// Web Push Notifications JavaScript
let registration = null;
let subscription = null;

// Check if browser supports service workers and push notifications
function isPushNotificationSupported() {
    return 'serviceWorker' in navigator && 'PushManager' in window;
}

// Register service worker
async function registerServiceWorker() {
    if (!isPushNotificationSupported()) {
        console.log('Push notifications are not supported in this browser');
        return false;
    }

    try {
        registration = await navigator.serviceWorker.register('/sw.js');
        console.log('Service Worker registered successfully');
        return true;
    } catch (error) {
        console.error('Service Worker registration failed:', error);
        return false;
    }
}

// Get VAPID public key from server
async function getVAPIDPublicKey() {
    try {
        const response = await fetch('/api/push/vapid-public-key');
        const data = await response.json();
        return data.publicKey;
    } catch (error) {
        console.error('Error fetching VAPID public key:', error);
        return null;
    }
}

// Convert VAPID public key to Uint8Array
function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
        .replace(/\-/g, '+')
        .replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
        outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
}

// Subscribe to push notifications
async function subscribeToPush() {
    if (!registration) {
        const registered = await registerServiceWorker();
        if (!registered) {
            return { success: false, error: 'Service Worker registration failed' };
        }
    }

    try {
        // Get VAPID public key
        const publicKey = await getVAPIDPublicKey();
        if (!publicKey) {
            return { success: false, error: 'VAPID public key not available' };
        }

        // Check current subscription
        subscription = await registration.pushManager.getSubscription();

        if (subscription) {
            console.log('Already subscribed to push notifications');
            return { success: true, subscription: subscription, message: 'Zaten abone oldunuz' };
        }

        // Subscribe to push notifications
        subscription = await registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: urlBase64ToUint8Array(publicKey)
        });

        // Send subscription to server
        const response = await fetch('/api/push/subscribe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(subscription.toJSON())
        });

        const data = await response.json();
        
        if (response.ok && data.success) {
            console.log('Successfully subscribed to push notifications');
            return { success: true, subscription: subscription, message: data.message || 'Bildirimler açıldı!' };
        } else {
            return { success: false, error: data.error || 'Subscription failed' };
        }

    } catch (error) {
        console.error('Error subscribing to push notifications:', error);
        
        if (error.name === 'NotAllowedError') {
            return { success: false, error: 'Bildirim izni reddedildi. Lütfen tarayıcı ayarlarından izin verin.' };
        }
        
        return { success: false, error: error.message || 'Abonelik hatası' };
    }
}

// Unsubscribe from push notifications
async function unsubscribeFromPush() {
    if (!registration) {
        return { success: false, error: 'Service Worker not registered' };
    }

    try {
        subscription = await registration.pushManager.getSubscription();

        if (!subscription) {
            return { success: false, error: 'Not subscribed to push notifications' };
        }

        // Unsubscribe
        const unsubscribed = await subscription.unsubscribe();
        
        if (unsubscribed) {
            // Notify server
            const response = await fetch('/api/push/unsubscribe', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(subscription.toJSON())
            });

            const data = await response.json();
            
            if (response.ok && data.success) {
                subscription = null;
                console.log('Successfully unsubscribed from push notifications');
                return { success: true, message: data.message || 'Bildirimler kapatıldı' };
            } else {
                return { success: false, error: data.error || 'Unsubscription failed' };
            }
        } else {
            return { success: false, error: 'Unsubscribe failed' };
        }

    } catch (error) {
        console.error('Error unsubscribing from push notifications:', error);
        return { success: false, error: error.message || 'Unsubscription error' };
    }
}

// Check subscription status
async function checkSubscriptionStatus() {
    if (!isPushNotificationSupported()) {
        return { supported: false, subscribed: false };
    }

    try {
        if (!registration) {
            const registered = await registerServiceWorker();
            if (!registered) {
                return { supported: true, subscribed: false, error: 'Service Worker not registered' };
            }
        }

        subscription = await registration.pushManager.getSubscription();
        return { 
            supported: true, 
            subscribed: subscription !== null,
            subscription: subscription 
        };
    } catch (error) {
        console.error('Error checking subscription status:', error);
        return { supported: true, subscribed: false, error: error.message };
    }
}

// Request notification permission and subscribe
async function requestNotificationPermission() {
    if (!('Notification' in window)) {
        return { success: false, error: 'Bu tarayıcı bildirimleri desteklemiyor' };
    }

    let permission = Notification.permission;

    if (permission === 'default') {
        permission = await Notification.requestPermission();
    }

    if (permission === 'granted') {
        return await subscribeToPush();
    } else if (permission === 'denied') {
        return { success: false, error: 'Bildirim izni reddedildi. Lütfen tarayıcı ayarlarından izin verin.' };
    } else {
        return { success: false, error: 'Bildirim izni alınamadı' };
    }
}

// Initialize push notifications on page load
async function initPushNotifications() {
    if (!isPushNotificationSupported()) {
        return;
    }

    // Register service worker
    await registerServiceWorker();

    // Check subscription status
    const status = await checkSubscriptionStatus();
    
    // Update UI based on status
    updateNotificationButton(status.subscribed);

    return status;
}

// Update notification button UI
function updateNotificationButton(subscribed) {
    const button = document.getElementById('push-notification-btn');
    if (!button) return;

    if (subscribed) {
        button.innerHTML = `
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            Bildirimler Açık
        `;
        button.classList.remove('bg-indigo-600', 'hover:bg-indigo-700');
        button.classList.add('bg-green-600', 'hover:bg-green-700');
        button.onclick = async () => {
            const result = await unsubscribeFromPush();
            if (result.success) {
                alert(result.message);
                updateNotificationButton(false);
            } else {
                alert('Hata: ' + result.error);
            }
        };
    } else {
        button.innerHTML = `
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            Bildirimleri Aç
        `;
        button.classList.remove('bg-green-600', 'hover:bg-green-700');
        button.classList.add('bg-indigo-600', 'hover:bg-indigo-700');
        button.onclick = async () => {
            const result = await requestNotificationPermission();
            if (result.success) {
                alert(result.message);
                updateNotificationButton(true);
            } else {
                alert('Hata: ' + result.error);
            }
        };
    }
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPushNotifications);
} else {
    initPushNotifications();
}

