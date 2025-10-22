// static/sw.js
const CACHE_VERSION = 'v4';
const CACHE_NAME = `darts-golf-${CACHE_VERSION}`;

// Core shell + icons/images to precache on first install
const CORE_ASSETS = [
  '/', // HTML shell
  '/static/manifest.webmanifest',
  '/static/style.css', // New: Externalized CSS

  // Icons / favicons used across the app
  '/static/icons/dartboard.svg',
  '/static/icons/dartboard-32.png',
  '/static/icons/dartboard-180.png',
  '/static/icons/dartboard-192.png',
  '/static/icons/dartboard-512.png',
  '/static/favicon.ico',

  // Hero / page images
  '/static/images/dartsgolf4.png',
  '/static/images/finalstanding1.png',
];

self.addEventListener('install', (event) => {
  // Take control ASAP on next step
  self.skipWaiting();

  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);
    // Add core assets, but don't fail install if one URL 404s
    await Promise.all( // Don't fail install if one URL 404s
      CORE_ASSETS.map((u) => cache.add(u).catch(() => void 0))
    );
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys.map((k) => (k.startsWith('darts-golf-') && k !== CACHE_NAME) ? caches.delete(k) : null)
    );
    await self.clients.claim();
  })());
});

// Fetch strategy
// - Ignore non-GET and /api/*
// - HTML navigations: network-first, fallback to cache (shell OK)
// - Same-origin images: cache-first (fast repeat loads / offline)
// - Same-origin other static: stale-while-revalidate
// - Cross-origin: passthrough (avoid opaque cache bloat)
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== 'GET') return;
  if (url.pathname.startsWith('/api/')) return;

  // HTML navigations
  const wantsHTML =
    request.mode === 'navigate' ||
    (request.headers.get('accept') || '').includes('text/html');

  if (wantsHTML) {
    event.respondWith((async () => {
      try {
        const network = await fetch(request);
        const cache = await caches.open(CACHE_NAME);
        cache.put(request, network.clone());
        return network;
      } catch {
        const cache = await caches.open(CACHE_NAME);
        return (await cache.match(request)) || (await cache.match('/'));
      }
    })());
    return;
  }

  // Images: cache-first for speed & offline
  const isImage =
    request.destination === 'image' ||
    (request.headers.get('accept') || '').includes('image/');

  if (isImage && url.origin === location.origin) {
    event.respondWith((async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);
      if (cached) return cached;

      try {
        const res = await fetch(request);
        if (res && res.ok) cache.put(request, res.clone());
        return res;
      } catch {
        // Optional fallback to a known image if desired
        const fallback = await cache.match('/static/images/dartsgolf4.png');
        return fallback || Response.error();
      }
    })());
    return;
  }

  // Same-origin static: stale-while-revalidate
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
      return cached || networkFetch || Response.error();
    })());
    return;
  }

  // Cross-origin: passthrough
});

// Optional: runtime precache hook from the page
// Example:
// navigator.serviceWorker.controller?.postMessage({ type: 'PRECACHE', urls: ['/static/images/extra.png'] });
self.addEventListener('message', (event) => {
  const data = event.data || {};
  if (data.type === 'PRECACHE' && Array.isArray(data.urls) && data.urls.length) {
    event.waitUntil((async () => {
      const cache = await caches.open(CACHE_NAME);
      // Only cache same-origin URLs
      const sameOrigin = data.urls.filter((u) => {
        try { return new URL(u, location.origin).origin === location.origin; }
        catch { return false; }
      });
      await Promise.all(sameOrigin.map((u) => cache.add(u).catch(() => void 0)));
    })());
  }
});
