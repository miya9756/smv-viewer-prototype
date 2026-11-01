// In-browser decode of a compressed SmoothMotionVector scene (no server, no GPU
// for decode). Unzips the .zip, parses the .npz keyframe, and reconstructs each
// GOP's keyframe gaussians (DC colour only — no SH). Motion (ffmpeg.wasm) is in
// decode_motion.js. Mirrors decoder/load_reference.py (keep the dequant math in sync).
import { unzipSync } from './vendor/fflate.js';

const SH_C0 = 0.28209479177387814;

// ---- .npy / .npz parsing ---------------------------------------------------
export function parseNpy(u8) {
  // header: \x93NUMPY, ver(2), headerlen(2 or 4), ascii dict, then raw C-order data
  const ver = u8[6];
  let hlen, hstart;
  if (ver === 1) { hlen = u8[8] | (u8[9] << 8); hstart = 10; }
  else { hlen = u8[8] | (u8[9] << 8) | (u8[10] << 16) | (u8[11] << 24); hstart = 12; }
  const hdr = new TextDecoder().decode(u8.subarray(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(hdr)[1];
  const shapeStr = /'shape':\s*\(([^)]*)\)/.exec(hdr)[1];
  const shape = shapeStr.split(',').map(s => s.trim()).filter(s => s.length).map(Number);
  const off = hstart + hlen;
  const buf = u8.buffer.slice(u8.byteOffset + off, u8.byteOffset + u8.byteLength);
  let data;
  if (descr === '<f2' || descr === '|f2') data = new Uint16Array(buf);   // float16 (raw)
  else if (descr === '<f4') data = new Float32Array(buf);
  else if (descr === '<f8') data = new Float64Array(buf);
  else if (descr === '|u1' || descr === '<u1') data = new Uint8Array(buf);
  else if (descr === '<u2') data = new Uint16Array(buf);
  else if (descr === '<i8') data = new BigInt64Array(buf);
  else if (descr === '|b1') data = new Uint8Array(buf);
  else throw new Error('npy dtype not handled: ' + descr);
  return { descr, shape, data };
}

export function parseNpz(u8) {
  const files = unzipSync(u8);
  const out = {};
  for (const name in files) out[name.replace(/\.npy$/, '')] = parseNpy(files[name]);
  return out;
}

// scalar value from a 0-d npy entry
const scalar = (e) => (e.data.length ? (typeof e.data[0] === 'bigint' ? Number(e.data[0]) : e.data[0]) : 0);

export function f16to32(u16, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = u16[i], s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    out[i] = e === 0 ? (s ? -1 : 1) * 2 ** -14 * (f / 1024)
      : e === 0x1f ? (f ? NaN : (s ? -1 : 1) * Infinity)
      : (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
  }
  return out;
}

// dequantize q -> float: q/(2^nbit-1) * range + min, with `nc` components broadcast
function dequant(qData, mn, rg, nbit, n, nc) {
  const max = 2 ** Number(nbit) - 1;
  const out = new Float32Array(n * nc);
  for (let i = 0; i < n; i++)
    for (let c = 0; c < nc; c++) {
      const k = i * nc + c;
      out[k] = (qData[k] / max) * rg[c] + mn[c];
    }
  return out;
}

// gather rows (nc-wide) by index list
function gather(src, idx, nc) {
  const out = new Float32Array(idx.length * nc);
  for (let i = 0; i < idx.length; i++)
    for (let c = 0; c < nc; c++) out[i * nc + c] = src[idx[i] * nc + c];
  return out;
}
function concat(a, b) { const o = new Float32Array(a.length + b.length); o.set(a); o.set(b, a.length); return o; }

// ---- per-GOP keyframe (DC colour) ------------------------------------------
function decodeGopAttrs(npz, g) {
  const pre = `gop_${g}_`;
  const n = npz[pre + 'position'].shape[0];
  const center = f16to32(npz[pre + 'position'].data, n * 3);
  const rot = dequant(npz[pre + 'rotation'].data, npz[pre + 'rotation_min'].data,
    npz[pre + 'rotation_range'].data, scalar(npz[pre + 'rotation_n_bit']), n, 4);   // min/range len 4
  // scaling: per-axis quantized but a single scalar min/range -> broadcast to 3
  const smn = npz[pre + 'scaling_min'].data[0], srg = npz[pre + 'scaling_range'].data[0];
  const scale = dequant(npz[pre + 'scaling'].data, [smn, smn, smn], [srg, srg, srg],
    scalar(npz[pre + 'scaling_n_bit']), n, 3);
  const opacity = dequant(npz[pre + 'opacity'].data, [npz[pre + 'opacity_min'].data[0]],
    [npz[pre + 'opacity_range'].data[0]], scalar(npz[pre + 'opacity_n_bit']), n, 1);
  const dc = dequant(npz[pre + 'base_color'].data, npz[pre + 'base_color_min'].data,
    npz[pre + 'base_color_range'].data, scalar(npz[pre + 'color_n_bit']), n, 3);       // min/range len 3
  return { n, center, rot, scale, opacity, dc };
}

function toRgba(dc, opacity, n) {
  const color = new Uint32Array(n);          // packed RGBA per splat (renderer expects this)
  const cb = new Uint8Array(color.buffer);
  const cl = (v) => v < 0 ? 0 : v > 255 ? 255 : v;
  for (let i = 0; i < n; i++) {
    cb[i*4]   = cl((0.5 + SH_C0 * dc[i*3])   * 255);
    cb[i*4+1] = cl((0.5 + SH_C0 * dc[i*3+1]) * 255);
    cb[i*4+2] = cl((0.5 + SH_C0 * dc[i*3+2]) * 255);
    cb[i*4+3] = cl(opacity[i] * 255);
  }
  return color;
}
function normQuats(q, n) {            // in place wxyz normalize
  for (let i = 0; i < n; i++) {
    const o = i*4, l = Math.hypot(q[o], q[o+1], q[o+2], q[o+3]) || 1;
    q[o]/=l; q[o+1]/=l; q[o+2]/=l; q[o+3]/=l;
  }
  return q;
}

// ---- view-dependent specular (degree-3 SH, codebook-compressed) -------------
// Each scene stores SH "rest" bands as a 1024-entry codebook (uint8) + a per-splat
// uint16 index. The static codebook is GOP-0-shared; each GOP also has a dynamic
// codebook. We hand the renderer, per keyframe: a combined [static|dynamic] codebook
// (Float32, 2048*45, coeff-major) + a per-splat index into it (static entry i -> i,
// dynamic entry j -> 1024+j). The shader evaluates the SH against the view direction.
const SH_HALF = 1024;             // codebook entries per half (static / dynamic)
// `key` is the full npz key ('gop_0_sh_codebook_static' or, single-GOP, 'sh_codebook').
function decodeCodebookKey(npz, key, nBitKey) {   // -> Float32Array(K*45) or null
  const e = npz[key];
  if (!e) return null;
  const q = e.data, mn = npz[`${key}_min`].data, rg = npz[`${key}_range`].data;
  const max = 2 ** Number(scalar(npz[nBitKey])) - 1;
  const K = e.shape[0], M = e.shape[1], out = new Float32Array(K * M);
  for (let k = 0; k < K; k++) { const o = k * M; for (let c = 0; c < M; c++) out[o + c] = q[o + c] / max * rg[c] + mn[c]; }
  return out;
}
const decodeCodebook = (npz, g, name) =>
  decodeCodebookKey(npz, `gop_${g}_${name}`, `gop_${g}_color_n_bit`);
function combineCB(staticCB, dynCB) {
  if (!staticCB || !dynCB) return null;
  const out = new Float32Array(staticCB.length + dynCB.length);   // 2048*45
  out.set(staticCB, 0); out.set(dynCB, staticCB.length);
  return out;
}

// ---- single-GOP archives (D-NeRF / nerfds) ---------------------------------
// These carry flat keys ('position', not 'gop_0_position'), one unsplit SH codebook
// that every splat indexes directly, and no mask_for_later. Mirrors the
// _is_single_gop_format() branch in decoder/load_reference.py. There is only ever a
// GOP 0, so nothing is ever carried into a later GOP (no static prefix, no dynamic_idx).
const isSingleGop = (npz) => !!npz['position'];

function decodeFlatAttrs(npz) {
  const n = npz['position'].shape[0];
  const center = f16to32(npz['position'].data, n * 3);
  const rot = dequant(npz['rotation'].data, npz['rotation_min'].data,
    npz['rotation_range'].data, scalar(npz['rotation_n_bit']), n, 4);
  const smn = npz['scaling_min'].data[0], srg = npz['scaling_range'].data[0];   // scalar -> broadcast to 3
  const scale = dequant(npz['scaling'].data, [smn, smn, smn], [srg, srg, srg],
    scalar(npz['scaling_n_bit']), n, 3);
  const opacity = dequant(npz['opacity'].data, [npz['opacity_min'].data[0]],
    [npz['opacity_range'].data[0]], scalar(npz['opacity_n_bit']), n, 1);
  const dc = dequant(npz['base_color'].data, npz['base_color_min'].data,
    npz['base_color_range'].data, scalar(npz['color_n_bit']), n, 3);
  return { n, center, rot, scale, opacity, dc };
}

function reconstructSingleGop(npz) {
  const a = decodeFlatAttrs(npz);
  normQuats(a.rot, a.n);
  const staticIdx = scalar(npz['static_idx']);          // dynamic splats are the tail [staticIdx, n)
  const kf = {
    center: a.center, scale: a.scale, quat: a.rot,
    color: toRgba(a.dc, a.opacity, a.n),
    numTotal: a.n, staticIdx, numDyn: a.n - staticIdx,
  };
  const cb = decodeCodebookKey(npz, 'sh_codebook', 'color_n_bit');
  if (cb) {                     // one codebook for every splat: index it directly, no SH_HALF offset
    kf.shCodebook = cb;
    kf.shIndex = npz['spec_color_indices'].data;
  }
  sanitizeKeyframe(kf);
  return kf;
}

// Build the GOP-0-shared context from the shared archive (gop_0_* keys). Returns the
// GOP-0 keyframe (.kf0) plus everything needed to reconstruct any later GOP's keyframe
// (the static-prefix source + SH codebook). For split streaming: fetch shared.npz first,
// then feed each on-demand gop_N.npz to reconstructGop().
export function reconstructShared(npz) {
  if (isSingleGop(npz)) {       // nothing to carry forward — GOP 0 is the whole scene
    const kf0 = reconstructSingleGop(npz);
    return { kf0, gop0: null, staticIdx0: kf0.staticIdx, prefixIdx: [],
             hasSH: !!kf0.shCodebook, staticCB: null, idxStatic0: null, singleGop: true };
  }
  const gop0 = decodeGopAttrs(npz, 0);
  normQuats(gop0.rot, gop0.n);
  const staticIdx0 = scalar(npz['gop_0_static_idx']);
  const kf0 = {
    // copy: sanitizeKeyframe(kf0) clamps floaters IN PLACE, and gop0.center is the
    // shared prefix source for every later GOP — mutating it would relocate GOP-0
    // floaters to the scene center (large scale + live alpha) and resurrect them as
    // big blobs in subsequent GOPs. Keep gop0.center raw so each GOP culls its own.
    center: gop0.center.slice(), scale: gop0.scale, quat: gop0.rot, color: toRgba(gop0.dc, gop0.opacity, gop0.n),
    numTotal: gop0.n, staticIdx: staticIdx0, numDyn: gop0.n - staticIdx0,
  };
  // static SH codebook is GOP-0-shared; null when a scene carries no specular layer
  const staticCB = decodeCodebook(npz, 0, 'sh_codebook_static');
  const hasSH = !!staticCB;
  const idxStatic0 = hasSH ? npz['gop_0_spec_color_indices_static'].data : null;
  if (hasSH) {
    const idxDyn0 = npz['gop_0_spec_color_indices_dynamic'].data, n0 = gop0.n;
    const si = new Uint16Array(n0);
    for (let i = 0; i < staticIdx0; i++) si[i] = idxStatic0[i];
    for (let d = 0; d < n0 - staticIdx0; d++) si[staticIdx0 + d] = SH_HALF + idxDyn0[d];
    kf0.shIndex = si;
    kf0.shCodebook = combineCB(staticCB, decodeCodebook(npz, 0, 'sh_codebook_dynamic'));
  }
  sanitizeKeyframe(kf0);
  // static-prefix indices: among GOP-0 static gaussians [0,staticIdx0), those with mask_for_later
  const mask = npz['gop_0_mask_for_later'].data, prefixIdx = [];   // Uint8 bool, len = staticIdx0
  for (let i = 0; i < staticIdx0; i++) if (mask[i]) prefixIdx.push(i);
  return { kf0, gop0, staticIdx0, prefixIdx, hasSH, staticCB, idxStatic0 };
}

// Reconstruct GOP g's keyframe (g>=1) from the shared context + that GOP's private archive
// (gop_g_* keys). gopNpz may be the dedicated gop_N.npz (split streaming) or the whole
// archive (legacy) — only gop_g_* keys are read from it.
export function reconstructGop(ctx, gopNpz, g) {
  if (ctx.singleGop) throw new Error(`single-GOP archive has no GOP ${g}`);
  const { gop0, prefixIdx, hasSH, staticCB, idxStatic0 } = ctx;
  const a = decodeGopAttrs(gopNpz, g);
  normQuats(a.rot, a.n);
  const center = concat(gather(gop0.center, prefixIdx, 3), a.center);
  const scale  = concat(gather(gop0.scale,  prefixIdx, 3), a.scale);
  const quat   = concat(gather(gop0.rot,    prefixIdx, 4), a.rot);
  const dc     = concat(gather(gop0.dc,     prefixIdx, 3), a.dc);
  const opac   = concat(gather(gop0.opacity,prefixIdx, 1), a.opacity);
  const numTotal = prefixIdx.length + a.n;
  const dynamicIdx = scalar(gopNpz[`gop_${g}_dynamic_idx`]);
  const kf = {
    center, scale, quat, color: toRgba(dc, opac, numTotal),
    numTotal, staticIdx: numTotal - dynamicIdx, numDyn: dynamicIdx,
  };
  if (hasSH) {
    // splat order = [prefix from GOP-0 static] ++ [this GOP's static] ++ [this GOP's dynamic]
    const isG = gopNpz[`gop_${g}_spec_color_indices_static`].data;    // len = a.n - dynamicIdx
    const idG = gopNpz[`gop_${g}_spec_color_indices_dynamic`].data;   // len = dynamicIdx
    const si = new Uint16Array(numTotal), staticCount = a.n - dynamicIdx;
    for (let k = 0; k < prefixIdx.length; k++) si[k] = idxStatic0[prefixIdx[k]];   // prefix uses static codebook
    for (let s = 0; s < staticCount; s++) si[prefixIdx.length + s] = isG[s];
    for (let d = 0; d < dynamicIdx; d++) si[prefixIdx.length + staticCount + d] = SH_HALF + idG[d];
    kf.shIndex = si;
    kf.shCodebook = combineCB(staticCB, decodeCodebook(gopNpz, g, 'sh_codebook_dynamic'));
  }
  sanitizeKeyframe(kf);              // remove extreme floater gaussians (fp16 overflow etc.)
  return kf;
}

// Legacy: reconstruct ALL GOP keyframes from one full archive (whole-zip loader / local dev).
export function reconstructKeyframes(npz) {
  const gopIds = Object.keys(npz).filter(k => /^gop_\d+_position$/.test(k))
    .map(k => +k.match(/^gop_(\d+)_/)[1]).sort((a, b) => a - b);
  const ctx = reconstructShared(npz);
  const out = [ctx.kf0];
  for (const g of gopIds.slice(1)) out.push(reconstructGop(ctx, npz, g));
  return out;
}

// A few gaussians per scene are extreme "floaters" (positions at tens of thousands
// of units, or non-finite from fp16 overflow). The CUDA rasterizer just culls them,
// but they wreck our CPU depth-sort range. Neutralize them: clamp to the scene
// center and make them invisible (alpha 0). One O(N) pass per GOP.
function sanitizeKeyframe(kf) {
  const c = kf.center, n = kf.numTotal, cb = new Uint8Array(kf.color.buffer);
  const step = Math.max(1, Math.floor(n / 4096));
  let sx = 0, sy = 0, sz = 0, m = 0;
  for (let i = 0; i < n; i += step) {
    const x = c[3*i], y = c[3*i+1], z = c[3*i+2];
    if (isFinite(x) && isFinite(y) && isFinite(z)) { sx += x; sy += y; sz += z; m++; }
  }
  if (!m) return 0;
  const cx = sx/m, cy = sy/m, cz = sz/m, ds = [];
  for (let i = 0; i < n; i += step) {
    const d = Math.hypot(c[3*i]-cx, c[3*i+1]-cy, c[3*i+2]-cz);
    if (isFinite(d)) ds.push(d);
  }
  ds.sort((a, b) => a - b);
  const thr = (ds[Math.floor(ds.length * 0.99)] || 1) * 20;   // 20x the 99th-pct extent
  let culled = 0;
  for (let i = 0; i < n; i++) {
    const x = c[3*i], y = c[3*i+1], z = c[3*i+2];
    if (!(isFinite(x) && isFinite(y) && isFinite(z)) || Math.hypot(x-cx, y-cy, z-cz) > thr) {
      c[3*i] = cx; c[3*i+1] = cy; c[3*i+2] = cz; cb[4*i+3] = 0; culled++;
    }
  }
  return culled;
}

export function unzipScene(u8) {
  return unzipSync(u8);   // {filename: Uint8Array}
}
