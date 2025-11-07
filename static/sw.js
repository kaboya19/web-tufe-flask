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

  // Generate unique tag if not provided (to ensure each notification is shown)
  const notificationTag = notificationData.tag || `web-tufe-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  
  event.waitUntil(
    self.registration.showNotification(notificationData.title, {
      body: notificationData.body,
      icon: notificationData.icon || '/static/icon-192x192.png',
      badge: notificationData.badge || '/static/badge-72x72.png',
      tag: notificationTag,
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

  let urlToOpen = event.notification.data?.url || '/';

  // Convert relative URL to absolute URL and handle encoding
  const origin = self.location.origin;
  
  if (urlToOpen.startsWith('/')) {
    // Encode the path part properly (handle spaces and special characters)
    try {
      // Split URL into path and query/hash parts
      const urlParts = urlToOpen.split('?');
      const pathPart = urlParts[0];
      const queryPart = urlParts[1] ? '?' + urlParts[1] : '';
      
      // Encode path segments properly
      const encodedPath = pathPart.split('/').map(segment => {
        // Don't encode the empty first segment or already encoded segments
        if (!segment || segment.includes('%')) return segment;
        return encodeURIComponent(segment);
      }).join('/');
      
      urlToOpen = origin + encodedPath + queryPart;
    } catch (e) {
      // Fallback: simple concatenation
      console.error('Error encoding URL:', e);
      urlToOpen = origin + urlToOpen;
    }
  } else if (!urlToOpen.startsWith('http://') && !urlToOpen.startsWith('https://')) {
    // If it's not a full URL and not starting with /, make it absolute
    urlToOpen = origin + '/' + urlToOpen;
  }

  console.log('Opening URL:', urlToOpen);

  event.waitUntil(
    clients.matchAll({
      type: 'window',
      includeUncontrolled: true
    }).then((clientList) => {
      // Check if there is already a window/tab open with the target URL
      // For PDF files or different URLs, always open new window
      const isPDF = urlToOpen.toLowerCase().endsWith('.pdf') || urlToOpen.includes('/pdf/');
      
      if (!isPDF) {
        for (let i = 0; i < clientList.length; i++) {
          const client = clientList[i];
          // Check if the URL matches (handle relative vs absolute)
          const clientUrl = new URL(client.url);
          const targetUrl = new URL(urlToOpen);
          if (clientUrl.pathname === targetUrl.pathname && 'focus' in client) {
            return client.focus();
          }
        }
      }
      
      // If not found or is PDF, open a new window/tab
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

