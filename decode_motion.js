// In-browser motion decode: ffmpeg.wasm decodes the 6 HEVC motion videos per GOP,
// then we dequantize + apply (xyz offsets + quaternion deltas) to the keyframe's
// dynamic block. Output is a Float32 array, [F*D*7] = (x,y,z, qw,qx,qy,qz) per
// dynamic gaussian per frame.
import { FFmpeg } from './vendor/ffmpeg/index.js';
import { toBlobURL } from './vendor/util/index.js';

let _ff = null, _loading = null, _coreURLs = null;
// Fetch the ~31 MB core+wasm exactly once and keep the blob URLs alive for the
// whole session. recreateFFmpeg() (called every few GOPs to reset the heap) then
// re-instantiates from these in-memory blobs with NO network round-trip — crucial
// on cellular, where re-fetching the wasm each reset caused multi-second stalls.
async function coreURLs() {
  if (_coreURLs) return _coreURLs;
  const base = './vendor/core';
  _coreURLs = {
    coreURL: await toBlobURL(`${base}/ffmpeg-core.js`, 'text/javascript'),
    wasmURL: await toBlobURL(`${base}/ffmpeg-core.wasm`, 'application/wasm'),
  };
  return _coreURLs;
}
export function ffmpegReady(onLog) {
  if (_ff) return Promise.resolve(_ff);
  if (_loading) return _loading;
  _loading = (async () => {
    const ff = new FFmpeg();
    if (onLog) ff.on('log', ({ message }) => onLog(message));
    await ff.load(await coreURLs());
    _ff = ff; return ff;
  })();
  // Never cache a FAILED load: drop the promise so the next call retries a fresh instance
  // (otherwise a one-off load error wedges the decoder permanently — every later call would
  // await the same rejected promise).
  _loading.catch(() => { _loading = null; });
  return _loading;
}

// Tear down and re-instantiate ffmpeg.wasm — resets its WASM heap so long decode
// runs don't accumulate memory (which otherwise causes "memory access out of bounds").
export async function recreateFFmpeg(onLog) {
  if (_ff) { try { _ff.terminate(); } catch (e) {} }
  _ff = null; _loading = null;
  return ffmpegReady(onLog);
}

const findFile = (files, suffix) => {
  const k = Object.keys(files).find(n => n.endsWith(suffix));
  return k ? files[k] : null;
};
const listMetas = (files, dir) =>
  Object.keys(files).filter(n => n.includes(dir) && n.endsWith('.json')).sort()
    .map(n => JSON.parse(new TextDecoder().decode(files[n])));

let _ctr = 0;
async function decodeVideo(ff, bytes) {     // .mkv -> Uint8Array rgb24 (F*D*3)
  const inN = `m${_ctr}.mkv`, outN = `m${_ctr}.rgb`; _ctr++;
  try {
    // writeFile lists data.buffer as transferable → it DETACHES the source ArrayBuffer.
    // The caller's segment bytes (gops[g].src) are cached for on-demand re-decode, so we
    // must hand ffmpeg a fresh copy; otherwise the 2nd decode of a GOP (loop/seek-back)
    // reads an emptied buffer and the decoder hangs.
    await ff.writeFile(inN, bytes.slice());
    await ff.exec(['-i', inN, '-f', 'rawvideo', '-pix_fmt', 'rgb24', outN]);
    return await ff.readFile(outN);
  } finally {                                // always clean up MEMFS, even on error
    try { await ff.deleteFile(inN); } catch (e) {}
    try { await ff.deleteFile(outN); } catch (e) {}
  }
}

// ── Dequant worker ───────────────────────────────────────────────────────────
// The per-frame dequant (an F·D quaternion-multiply loop) used to run on the main
// thread and stalled the render loop at every GOP boundary. We hand it to a
// dedicated Web Worker instead; decoded RGB frames go over as transferables
// (zero-copy) and the finished `motion` buffer comes back the same way.
let _dq = null, _dqJobs = null, _dqId = 0;
function dqWorker() {
  if (_dq) return _dq;
  _dqJobs = new Map();
  _dq = new Worker(new URL('./dequant_worker.js', import.meta.url));
  _dq.onmessage = (e) => {
    const job = _dqJobs.get(e.data.id);
    if (job) { _dqJobs.delete(e.data.id); job.resolve(e.data.motion); }
  };
  _dq.onerror = (err) => {                    // worker died → fail pending jobs and drop it so the next call respawns
    const e = new Error('dequant worker error: ' + ((err && err.message) || err));
    for (const job of _dqJobs.values()) job.reject(e);
    _dqJobs.clear(); try { _dq.terminate(); } catch (_) {} _dq = null;
  };
  return _dq;
}
function dequantInWorker(payload, transfers) {
  const w = dqWorker(), id = ++_dqId;
  return new Promise((resolve, reject) => {
    _dqJobs.set(id, { resolve, reject });
    w.postMessage({ id, ...payload }, transfers);
  });
}

// ── Per-video decode pool (primary + helpers) ────────────────────────────────
// ffmpeg.wasm is single-threaded (one exec per instance at a time), so a GOP's 5-6 motion videos
// used to decode strictly serially on the one instance — the main decode-throughput ceiling. Here
// we fan them out across the caller's `primary` instance plus a few lazily-spawned HELPER instances
// (loaded from the same in-memory wasm blobs — no extra network fetch). Each instance still runs one
// video at a time; the videos just spread across instances. POOL===1, a low-memory device, or a
// helper that fails to spawn all degrade transparently to the original serial-on-primary path.
const _hc = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
const _lowMem = (typeof navigator !== 'undefined' && navigator.deviceMemory && navigator.deviceMemory < 4);
const POOL = _lowMem ? 1 : Math.max(1, Math.min(3, Math.floor(_hc / 2)));   // total concurrent decoders (primary + helpers)
const HELPER_RESET = 12;                    // recreate a helper's wasm heap after this many decodes (mirrors the primary's periodic reset)
const _helpers = [];                        // persistent helper instances: { ff, count }
let _helpersInit = null;

async function _makeHelper() { const ff = new FFmpeg(); await ff.load(await coreURLs()); return { ff, count: 0 }; }
function _ensureHelpers() {                  // spawn POOL-1 helpers once, best-effort (non-blocking for the caller)
  if (_helpersInit) return _helpersInit;
  _helpersInit = (async () => {
    for (let i = _helpers.length; i < POOL - 1; i++) {
      try { _helpers.push(await _makeHelper()); }
      catch (e) { console.warn('[smv] ffmpeg helper', i, 'spawn failed — continuing with fewer:', (e && e.message) || e); break; }
    }
    console.info(`[smv] parallel decode: ${1 + _helpers.length}/${POOL} ffmpeg instance(s) (hardwareConcurrency=${_hc})`);
  })();
  return _helpersInit;
}
async function _recreateHelper(slot) { try { slot.ff.terminate(); } catch (e) {} slot.ff = new FFmpeg(); await slot.ff.load(await coreURLs()); slot.count = 0; }

// Decode `srcs` (video byte buffers) concurrently across the pool; returns rgb24 arrays in the SAME
// order. The primary is reset externally (by manageBuffer); helpers self-reset by decode count.
async function poolDecode(primary, srcs) {
  _ensureHelpers();                                              // kick off helper spawn in the background — the first GOP runs on the primary alone (no added time-to-first-frame)
  const slots = [{ ff: primary, primary: true }, ..._helpers];  // whatever helpers are ready right now
  const out = new Array(srcs.length);
  let next = 0;                                                  // shared cursor; `const i = next++` runs sync before any await, so no two workers grab the same index
  await Promise.all(slots.map(async (slot) => {
    for (;;) {
      const i = next++;
      if (i >= srcs.length) break;
      if (!slot.primary && slot.count >= HELPER_RESET) { try { await _recreateHelper(slot); } catch (e) {} }
      try { out[i] = await decodeVideo(slot.ff, srcs[i]); }
      catch (e) {
        if (slot.primary) throw e;                               // let manageBuffer's handler recreate the primary + retry the GOP
        console.warn('[smv] ffmpeg helper decode failed, recreating heap + retrying once:', (e && e.message) || e);
        await _recreateHelper(slot); out[i] = await decodeVideo(slot.ff, srcs[i]);
      }
      if (!slot.primary) slot.count++;
    }
  }));
  return out;
}

// Decode one GOP's motion -> { motion: Float32[F*D*7], F, D }
export async function decodeMotionGop(ff, files, gop, keyframe) {
  const D = keyframe.numDyn, si = keyframe.staticIdx;
  const baseXyz = keyframe.center.subarray(si * 3);     // D*3
  const baseRot = keyframe.quat.subarray(si * 4);       // D*4
  const xu = findFile(files, `gop_${gop}_xyz_offsets_u.mkv`);
  const xl = findFile(files, `gop_${gop}_xyz_offsets_l.mkv`);   // may be absent
  const vw = findFile(files, `gop_${gop}_quat_w.mkv`);
  const vx = findFile(files, `gop_${gop}_quat_x.mkv`);
  const vy = findFile(files, `gop_${gop}_quat_y.mkv`);
  const vz = findFile(files, `gop_${gop}_quat_z.mkv`);
  const mX = listMetas(files, `gop_${gop}_motion_vectors_meta/xyz_offsets/`);
  const mW = listMetas(files, `gop_${gop}_motion_vectors_meta/q_w/`);
  const mXr = listMetas(files, `gop_${gop}_motion_vectors_meta/q_x/`);
  const mY = listMetas(files, `gop_${gop}_motion_vectors_meta/q_y/`);
  const mZ = listMetas(files, `gop_${gop}_motion_vectors_meta/q_z/`);

  const _t0 = performance.now();
  // Decode this GOP's videos concurrently across the pool (primary `ff` + helpers) rather than
  // serially. Order must match the readback below; xl (lower byte) is optional so it goes last.
  const srcs = [xu, vw, vx, vy, vz];
  if (xl) srcs.push(xl);
  const dec = await poolDecode(ff, srcs);
  const uD = dec[0], wD = dec[1], xD = dec[2], yD = dec[3], zD = dec[4];
  const lD = xl ? dec[5] : null;
  const F = uD.length / (D * 3);

  // Offload the heavy F·D dequant loop to the worker. baseXyz/baseRot go as COPIES
  // (the originals stay live on the main thread for rendering); the RGB frames are
  // transferred (zero-copy — we don't need them on the main thread after this).
  const bx = baseXyz.slice(), br = baseRot.slice();
  const transfers = [uD.buffer, wD.buffer, xD.buffer, yD.buffer, zD.buffer, bx.buffer, br.buffer];
  if (lD) transfers.push(lD.buffer);
  const motion = await dequantInWorker(
    { D, F, baseXyz: bx, baseRot: br, uD, lD, wD, xD, yD, zD, mX, mW, mXr, mY, mZ },
    transfers,
  );
  // decodeMs lets the UI report per-GOP decode cost. We deliberately do NOT
  // retain the decoded RGB frames: the colorized motion-vector image is
  // re-derived on the fly from `motion` + the keyframe (see paintMV in
  // index.html), so nothing extra is held in memory per GOP.
  return { motion, F, D, decodeMs: performance.now() - _t0 };
}
