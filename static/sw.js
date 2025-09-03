const CACHE = 'darts-golf-v1';
const ASSETS = [
  '/', // html shell
  '/static/manifest.webmanifest',
  '/static/icons/dartboard-192.png',
  '/static/icons/dartboard-512.png',
  '/static/icons/dartboard.svg',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  const { request } = e;
  // network-first for HTML to avoid stale game states, cache-first for others
  if (request.mode === 'navigate') {
    e.respondWith(
      fetch(request).then((resp) => {
        const copy = resp.clone();
        caches.open(CACHE).then((c) => c.put(request, copy));
        return resp;
      }).catch(() => caches.match(request))
    );
  } else {
    e.respondWith(
      caches.match(request).then((cached) =>
        cached ||
        fetch(request).then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(request, copy));
          return resp;
        }).catch(() => cached)
      )
    );
  }
});
