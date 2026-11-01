// Service worker for the SmoothMotionVectors web player.
//
// Its one job: make the vendored, immutable assets — above all the ~31 MB
// ffmpeg.wasm decoder core — survive across page loads AND across the browser's
// own HTTP-cache evictions. On cellular, mobile browsers evict aggressively, so
// without this the decoder gets re-downloaded on a fresh visit (and, before the
// blob-URL fix, on every in-session reset), stalling motion playback for seconds.
//
// Strategy: cache-first for everything under /vendor/ (third-party, content never
// changes). The player's OWN code (index.html, decode*.js, scenes.json) and the
// scene .zips are deliberately NOT handled here, so they always hit the network
// and never go stale after a redeploy.
const VERSION = 'smv-vendor-v1';
const PRECACHE = [
  './vendor/core/ffmpeg-core.js',
  './vendor/core/ffmpeg-core.wasm',
];

self.addEventListener('install', (e) => {
  self.skipWaiting();
  // Best-effort precache of the heavy core so even the first reset is instant.
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(PRECACHE)).catch(() => {}));
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;        // never touch cross-origin (fonts, etc.)
  if (!url.pathname.includes('/vendor/')) return;    // only immutable vendored assets
  e.respondWith((async () => {
    const cache = await caches.open(VERSION);
    const hit = await cache.match(req);
    if (hit) return hit;                             // served from cache — zero network
    try {
      const res = await fetch(req);
      if (res.ok) cache.put(req, res.clone());       // populate on first miss
      return res;
    } catch (err) {
      const fallback = await cache.match(req);        // offline: last resort
      if (fallback) return fallback;
      throw err;
    }
  })());
});
