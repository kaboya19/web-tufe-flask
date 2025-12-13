# Render'da Site Çöktükten Sonra Geri Getirme Rehberi

## 🚨 Acil Durum: Site Çöktü

### 1. Render Dashboard'dan Manuel Restart

1. **Render Dashboard'a gidin**: https://dashboard.render.com
2. **Servisinizi seçin**: `web-tufe-flask`
3. **"Manual Deploy"** veya **"Restart"** butonuna tıklayın
4. **Log'ları kontrol edin**: Hata mesajlarını görmek için "Logs" sekmesine bakın

### 2. Health Check Endpoint

Site artık `/health` endpoint'i ile kontrol edilebilir:
- **URL**: `https://your-site.onrender.com/health`
- **Sağlıklı**: `{"status": "healthy", ...}` döner
- **Sorunlu**: `{"status": "unhealthy", "error": "..."}` döner

### 3. Log'ları İnceleme

Render Dashboard'da:
1. **Logs** sekmesine gidin
2. **Son hataları** kontrol edin
3. **"Error"** veya **"Exception"** kelimelerini arayın

### 4. Otomatik Restart Ayarları

Render otomatik olarak şunları yapar:
- **Health check başarısız olursa** → Otomatik restart
- **Memory limit aşılırsa** → Otomatik restart
- **Crash olursa** → Otomatik restart

### 5. Yaygın Sorunlar ve Çözümleri

#### Memory Limit Aşımı
**Belirtiler**: 
- Site yavaş çalışıyor
- "Out of memory" hatası

**Çözüm**:
- Render planınızı yükseltin (daha fazla RAM)
- Cache timeout'unu azaltın (app.py'de `CACHE_DEFAULT_TIMEOUT`)
- Worker sayısını azaltın (render.yaml'da)

#### Timeout Hatası
**Belirtiler**:
- İstekler zaman aşımına uğruyor
- 504 Gateway Timeout

**Çözüm**:
- Timeout süresini artırın (zaten 120 saniye)
- Yavaş işlemleri optimize edin
- Database sorgularını optimize edin

#### CSV Dosya Okuma Hatası
**Belirtiler**:
- "File not found" hatası
- CSV dosyaları eksik

**Çözüm**:
- CSV dosyalarının repo'da olduğundan emin olun
- Git'e commit edildiğinden emin olun
- Render'ın dosya sistemine yüklendiğinden emin olun

### 6. Monitoring ve Alerting

Render Dashboard'da:
1. **Metrics** sekmesine gidin
2. **CPU, Memory, Response Time** grafiklerini izleyin
3. **Alerts** ayarlayın (e-posta bildirimleri için)

### 7. Cache Temizleme

Eğer cache sorunluysa:
```python
# app.py'de cache'i temizlemek için:
cache.clear()
```

Veya Render'da environment variable ekleyin:
- `CLEAR_CACHE=true` → Uygulama başlarken cache temizlenir

### 8. Acil Restart Komutu (Render CLI)

Eğer Render CLI kuruluysa:
```bash
render services:restart web-tufe-flask
```

### 9. Rollback (Geri Alma)

Eğer yeni deploy sorun çıkarırsa:
1. **Deploys** sekmesine gidin
2. **Önceki başarılı deploy'u** seçin
3. **"Rollback"** butonuna tıklayın

### 10. Destek Alma

Render Support:
- **Email**: support@render.com
- **Documentation**: https://render.com/docs
- **Status Page**: https://status.render.com

## 🔧 Önleyici Önlemler

### 1. Health Check Monitoring
Render'da health check URL'ini ayarlayın:
- **Health Check Path**: `/health`
- **Check Interval**: 60 saniye

### 2. Resource Limits
Render planınızın limitlerini kontrol edin:
- **Free Plan**: 512 MB RAM, 100 GB bandwidth
- **Starter Plan**: 512 MB RAM, 100 GB bandwidth
- **Professional Plan**: Daha fazla kaynak

### 3. Log Retention
Render log'ları 7 gün saklar. Önemli hataları kaydedin.

### 4. Backup Stratejisi
- CSV dosyalarını düzenli yedekleyin
- Database'i (varsa) düzenli yedekleyin
- Git repo'yu düzenli push edin

## 📊 Performans İzleme

Render Dashboard'da şunları izleyin:
- **Response Time**: < 2 saniye olmalı
- **Memory Usage**: %80'in altında olmalı
- **CPU Usage**: Sürekli %100 olmamalı
- **Error Rate**: %1'in altında olmalı

## 🚀 Hızlı Restart Komutları

### Render Dashboard'dan:
1. Servis → Manual Deploy → Deploy

### Git Push ile:
```bash
git commit --allow-empty -m "Trigger redeploy"
git push
```

Bu, Render'ı yeni bir deploy tetikler ve siteyi yeniden başlatır.

