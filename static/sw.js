// Service Worker for Web Push Notifications
const CACHE_NAME = 'web-tufe-v1';
const STATIC_CACHE_URLS = [
  '/',
  '/static/sw.js'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('Service Worker installing...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Service Worker caching static assets');
        return cache.addAll(STATIC_CACHE_URLS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Push event - handle incoming push notifications
self.addEventListener('push', (event) => {
  console.log('Push notification received:', event);
  
  let notificationData = {
    title: 'Web TÜFE',
    body: 'Yeni bülten yayınlandı!',
    icon: '/static/icon-192x192.png',
    badge: '/static/badge-72x72.png',
    tag: 'web-tufe-notification',
    requireInteraction: false,
    data: {
      url: '/bultenler'
    }
  };

  if (event.data) {
    try {
      const data = event.data.json();
      notificationData = {
        ...notificationData,
        ...data
      };
    } catch (e) {
      notificationData.body = event.data.text();
    }
  }

  event.waitUntil(
    self.registration.showNotification(notificationData.title, {
      body: notificationData.body,
      icon: notificationData.icon || '/static/icon-192x192.png',
      badge: notificationData.badge || '/static/badge-72x72.png',
      tag: notificationData.tag || 'web-tufe-notification',
      requireInteraction: notificationData.requireInteraction || false,
      data: notificationData.data || { url: '/' },
      actions: notificationData.actions || [
        {
          action: 'open',
          title: 'Aç'
        },
        {
          action: 'close',
          title: 'Kapat'
        }
      ],
      vibrate: [200, 100, 200],
      timestamp: Date.now()
    })
  );
});

// Notification click event
self.addEventListener('notificationclick', (event) => {
  console.log('Notification clicked:', event);
  
  event.notification.close();

  if (event.action === 'close') {
    return;
  }

  const urlToOpen = event.notification.data?.url || '/';

  event.waitUntil(
    clients.matchAll({
      type: 'window',
      includeUncontrolled: true
    }).then((clientList) => {
      // Check if there is already a window/tab open with the target URL
      for (let i = 0; i < clientList.length; i++) {
        const client = clientList[i];
        if (client.url === urlToOpen && 'focus' in client) {
          return client.focus();
        }
      }
      // If not, open a new window/tab
      if (clients.openWindow) {
        return clients.openWindow(urlToOpen);
      }
    })
  );
});

// Background sync event (for future use)
self.addEventListener('sync', (event) => {
  console.log('Background sync:', event.tag);
  if (event.tag === 'sync-subscriptions') {
    event.waitUntil(syncSubscriptions());
  }
});

async function syncSubscriptions() {
  // Future: Sync subscription status with server
  console.log('Syncing subscriptions...');
}

