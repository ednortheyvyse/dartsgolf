// static/sw.js
const CACHE_NAME = 'darts-golf-v2';
const CORE_ASSETS = [
  '/', // html shell (index)
  '/static/manifest.webmanifest',
  '/static/icons/dartboard-192.png',
  '/static/icons/dartboard-512.png',
  '/static/icons/dartboard.svg',
  '/static/images/dartsgolf4.png',
  '/static/images/finalstanding1.png'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS))
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys.map((k) => (k !== CACHE_NAME ? caches.delete(k) : null))
    );
    await self.clients.claim();
  })());
});

// Strategy:
// - Ignore non-GET and /api/* (let app handle live game state).
// - HTML navigations: network-first, fallback to cache (stale shell OK).
// - Same-origin static: stale-while-revalidate.
// - Cross-origin: pass-through network (no opaque cache bloat).
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only GET requests are cacheable
  if (request.method !== 'GET') return;

  // Bypass API routes entirely (keep server-authoritative)
  if (url.pathname.startsWith('/api/')) return;

  // Handle navigations (HTML documents)
  const wantsHTML =
    request.mode === 'navigate' ||
    (request.headers.get('accept') || '').includes('text/html');

  if (wantsHTML) {
    event.respondWith((async () => {
      try {
        const network = await fetch(request);
        // Update cache with fresh copy
        const cache = await caches.open(CACHE_NAME);
        cache.put(request, network.clone());
        return network;
      } catch {
        // Offline fallback: try cache, then shell "/"
        const cache = await caches.open(CACHE_NAME);
        return (await cache.match(request)) || (await cache.match('/'));
      }
    })());
    return;
  }

  // Same-origin static assets: stale-while-revalidate
  if (url.origin === location.origin) {
    event.respondWith((async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);
      const networkFetch = fetch(request)
        .then((res) => {
          if (res && res.ok) cache.put(request, res.clone());
          return res;
        })
        .catch(() => null);
      return cached || networkFetch;
    })());
    return;
  }

  // Cross-origin: just go to network
  // (If you want to cache a CDN, whitelist here.)
});
