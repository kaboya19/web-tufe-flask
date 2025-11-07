// Web Push Notifications JavaScript
let registration = null;
let subscription = null;

// Check if browser supports service workers and push notifications
function isPushNotificationSupported() {
    // Check for service worker support
    if (!('serviceWorker' in navigator)) {
        return false;
    }
    
    // Check for push manager support
    if (!('PushManager' in window)) {
        return false;
    }
    
    // iOS Safari check (requires iOS 16.4+)
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
    
    if (isIOS && isSafari) {
        // iOS 16.4+ support check
        const iOSVersion = navigator.userAgent.match(/OS (\d+)_(\d+)/);
        if (iOSVersion) {
            const major = parseInt(iOSVersion[1]);
            const minor = parseInt(iOSVersion[2]);
            // iOS 16.4+ required for web push
            if (major < 16 || (major === 16 && minor < 4)) {
                console.log('iOS version too old for push notifications (requires iOS 16.4+)');
                return false;
            }
        }
    }
    
    return true;
}

// Register service worker
async function registerServiceWorker() {
    if (!isPushNotificationSupported()) {
        console.log('Push notifications are not supported in this browser');
        return false;
    }

    try {
        // Register service worker with proper scope for mobile devices
        registration = await navigator.serviceWorker.register('/sw.js', {
            scope: '/'
        });
        
        console.log('Service Worker registered successfully', registration);
        
        // Wait for service worker to be ready (important for mobile)
        if (registration.installing) {
            console.log('Service Worker installing...');
            await new Promise((resolve) => {
                registration.installing.addEventListener('statechange', () => {
                    if (registration.installing.state === 'installed') {
                        resolve();
                    }
                });
            });
        } else if (registration.waiting) {
            console.log('Service Worker waiting...');
        } else if (registration.active) {
            console.log('Service Worker active');
        }
        
        return true;
    } catch (error) {
        console.error('Service Worker registration failed:', error);
        
        // Provide helpful error messages for mobile users
        if (error.message.includes('not allowed')) {
            console.error('Service Worker registration not allowed. Make sure you are on HTTPS.');
        } else if (error.message.includes('scope')) {
            console.error('Service Worker scope error. Check manifest.json scope setting.');
        }
        
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

    // Check if we're on a mobile device
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;

    let permission = Notification.permission;

    // For mobile devices, especially iOS, we need to request permission explicitly
    if (permission === 'default') {
        // On mobile, especially iOS, permission must be requested from a user gesture
        try {
            permission = await Notification.requestPermission();
        } catch (error) {
            console.error('Error requesting notification permission:', error);
            if (isIOS) {
                return { 
                    success: false, 
                    error: 'iOS\'ta bildirimler için lütfen "Ana Ekrana Ekle" yapın ve ayarlardan izin verin.' 
                };
            }
            return { success: false, error: 'Bildirim izni alınamadı: ' + error.message };
        }
    }

    if (permission === 'granted') {
        // Ensure service worker is registered before subscribing
        if (!registration) {
            const registered = await registerServiceWorker();
            if (!registered) {
                return { success: false, error: 'Service Worker kaydı başarısız oldu' };
            }
        }
        
        return await subscribeToPush();
    } else if (permission === 'denied') {
        let errorMessage = 'Bildirim izni reddedildi. ';
        if (isMobile) {
            if (isIOS) {
                errorMessage += 'iOS\'ta Ayarlar > Safari > Bildirimler\'den izin verebilirsiniz.';
            } else {
                errorMessage += 'Tarayıcı ayarlarından bildirim izni verebilirsiniz.';
            }
        } else {
            errorMessage += 'Tarayıcı ayarlarından izin verebilirsiniz.';
        }
        return { success: false, error: errorMessage };
    } else {
        return { success: false, error: 'Bildirim izni alınamadı' };
    }
}

// Initialize push notifications on page load
async function initPushNotifications() {
    if (!isPushNotificationSupported()) {
        // Hide button or show unsupported message
        const button = document.getElementById('push-notification-btn');
        if (button) {
            const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
            const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
            
            if (isIOS && isSafari) {
                button.innerHTML = `
                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                    iOS 16.4+ Gerekli
                `;
                button.disabled = true;
                button.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                button.style.display = 'none';
            }
        }
        return;
    }

    // Wait a bit for page to fully load (important for mobile)
    if (document.readyState === 'loading') {
        await new Promise(resolve => {
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', resolve);
            } else {
                resolve();
            }
        });
    }

    // Small delay for mobile devices to ensure everything is ready
    await new Promise(resolve => setTimeout(resolve, 100));

    // Register service worker
    const registered = await registerServiceWorker();
    if (!registered) {
        console.warn('Service Worker registration failed, but continuing...');
    }

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
        button.classList.remove('bg-gradient-to-r', 'from-purple-600', 'to-pink-600', 'hover:from-purple-700', 'hover:to-pink-700');
        button.classList.add('bg-gradient-to-r', 'from-green-600', 'to-emerald-600', 'hover:from-green-700', 'hover:to-emerald-700');
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
        button.classList.remove('bg-gradient-to-r', 'from-green-600', 'to-emerald-600', 'hover:from-green-700', 'hover:to-emerald-700');
        button.classList.add('bg-gradient-to-r', 'from-purple-600', 'to-pink-600', 'hover:from-purple-700', 'hover:to-pink-700');
        button.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            // Show loading state
            const originalHTML = button.innerHTML;
            button.innerHTML = `
                <svg class="animate-spin h-5 w-5 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Yükleniyor...
            `;
            button.disabled = true;
            
            try {
                const result = await requestNotificationPermission();
                if (result.success) {
                    // Use a more mobile-friendly notification method
                    showNotificationMessage(result.message, 'success');
                    updateNotificationButton(true);
                } else {
                    showNotificationMessage('Hata: ' + result.error, 'error');
                }
            } catch (error) {
                console.error('Error in button click handler:', error);
                showNotificationMessage('Bir hata oluştu: ' + error.message, 'error');
            } finally {
                button.disabled = false;
            }
        };
    }
}

// Show notification message (mobile-friendly)
function showNotificationMessage(message, type = 'info') {
    // Try to use a more user-friendly notification method
    // On mobile, alerts can be intrusive, so we'll use a toast-style notification
    
    // Create a toast element
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: ${type === 'success' ? '#10B981' : type === 'error' ? '#EF4444' : '#3B82F6'};
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 10000;
        font-weight: 500;
        max-width: 90%;
        text-align: center;
        animation: slideDown 0.3s ease-out;
    `;
    toast.textContent = message;
    
    // Add animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateX(-50%) translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }
        }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(toast);
    
    // Remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = 'slideDown 0.3s ease-out reverse';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
            if (style.parentNode) {
                style.parentNode.removeChild(style);
            }
        }, 300);
    }, 4000);
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Small delay for mobile to ensure everything is ready
        setTimeout(initPushNotifications, 200);
    });
} else {
    // Small delay for mobile to ensure everything is ready
    setTimeout(initPushNotifications, 200);
}

