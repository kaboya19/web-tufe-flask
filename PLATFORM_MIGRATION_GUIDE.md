# Platform Geçiş Rehberi

## 🚀 Railway'a Geçiş (Önerilen)

### Avantajlar
- ✅ Render'dan daha iyi performans
- ✅ Otomatik scaling
- ✅ Kolay geçiş
- ✅ $5 ücretsiz kredi/ay

### Adımlar

1. **Railway hesabı oluşturun**
   - https://railway.app
   - GitHub ile giriş yapın

2. **Yeni proje oluşturun**
   - New Project → Deploy from GitHub repo
   - Repo'nuzu seçin

3. **Environment Variables ekleyin**
   - Railway Dashboard → Variables
   - Render'daki tüm env var'ları ekleyin:
     - `GOOGLE_CREDENTIALS_BASE64`
     - `GOOGLE_CREDENTIALS_2_BASE64`
     - `SMTP_PASSWORD`
     - Diğerleri...

4. **Deploy**
   - Railway otomatik olarak `Procfile` veya `railway.json` dosyasını okur
   - İlk deploy otomatik başlar

5. **Custom Domain (opsiyonel)**
   - Railway Dashboard → Settings → Domains
   - Custom domain ekleyin

### Dosyalar
- `Procfile` - Railway otomatik okur
- `railway.json` - Gelişmiş yapılandırma için

---

## 🛫 Fly.io'ya Geçiş

### Avantajlar
- ✅ Edge deployment (dünya çapında hızlı)
- ✅ Ücretsiz tier: 3 VM
- ✅ Çok iyi performans

### Adımlar

1. **Fly.io CLI kurulumu**
   ```bash
   # Windows (PowerShell)
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   
   # Mac/Linux
   curl -L https://fly.io/install.sh | sh
   ```

2. **Giriş yapın**
   ```bash
   fly auth login
   ```

3. **Proje oluşturun**
   ```bash
   fly launch
   ```
   - `fly.toml` dosyası oluşturulur

4. **Deploy**
   ```bash
   fly deploy
   ```

5. **Environment Variables**
   ```bash
   fly secrets set GOOGLE_CREDENTIALS_BASE64="..."
   fly secrets set SMTP_PASSWORD="..."
   ```

### Dosyalar
- `fly.toml` - Fly.io yapılandırması (örnek: `fly.toml.example`)

---

## 🌊 DigitalOcean App Platform'a Geçiş

### Avantajlar
- ✅ Basit ve güvenilir
- ✅ İyi performans
- ✅ Makul fiyat ($5/ay)

### Adımlar

1. **DigitalOcean hesabı oluşturun**
   - https://cloud.digitalocean.com

2. **App Platform'a gidin**
   - Create → App Platform
   - GitHub repo'nuzu bağlayın

3. **Yapılandırma**
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `gunicorn app:app --workers 4 --threads 2 --timeout 120 --worker-class sync --bind 0.0.0.0:$PORT`
   - Health Check: `/health`

4. **Environment Variables**
   - Settings → App-Level Environment Variables
   - Tüm env var'ları ekleyin

5. **Deploy**
   - Create Resources → Deploy

---

## 🐍 PythonAnywhere'a Geçiş

### Avantajlar
- ✅ Python odaklı
- ✅ Basit kurulum
- ✅ CSV dosyaları için uygun

### Adımlar

1. **Hesap oluşturun**
   - https://www.pythonanywhere.com
   - Free tier ile başlayın

2. **Web app oluşturun**
   - Web tab → Add a new web app
   - Flask seçin
   - Python 3.11 seçin

3. **Dosyaları yükleyin**
   - Files tab → Upload files
   - Tüm dosyaları yükleyin

4. **WSGI dosyası düzenleyin**
   - Web tab → WSGI configuration file
   - `app.py`'yi import edin

5. **Environment Variables**
   - Web tab → Environment variables
   - Tüm env var'ları ekleyin

6. **Reload**
   - Web tab → Reload

---

## 📊 Platform Karşılaştırması

| Özellik | Railway | Fly.io | DigitalOcean | PythonAnywhere |
|---------|---------|--------|--------------|----------------|
| **Ücretsiz Tier** | $5 kredi/ay | 3 VM | $5/ay | Sınırlı |
| **Performans** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Kolaylık** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Scaling** | Otomatik | Otomatik | Manuel | Manuel |
| **Edge Deployment** | ❌ | ✅ | ❌ | ❌ |
| **CSV Desteği** | ✅ | ✅ | ✅ | ✅ |

---

## 🎯 Hangi Platformu Seçmeliyim?

### Railway Önerilir Eğer:
- ✅ Render'dan kolay geçiş istiyorsanız
- ✅ Otomatik scaling istiyorsanız
- ✅ En iyi performans istiyorsanız
- ✅ Ücretsiz tier yeterli ise

### Fly.io Önerilir Eğer:
- ✅ Global edge deployment istiyorsanız
- ✅ En yüksek performans istiyorsanız
- ✅ CLI kullanımından rahatsız değilseniz

### DigitalOcean Önerilir Eğer:
- ✅ Basit ve güvenilir platform istiyorsanız
- ✅ $5/ay ödemeye hazırsanız
- ✅ Manuel kontrol istiyorsanız

### PythonAnywhere Önerilir Eğer:
- ✅ Python odaklı platform istiyorsanız
- ✅ En basit kurulum istiyorsanız
- ✅ CSV dosyaları için özel ihtiyaç varsa

---

## ⚠️ Geçiş Öncesi Kontrol Listesi

- [ ] Tüm environment variables listesi hazır
- [ ] CSV dosyaları repo'da
- [ ] `requirements.txt` güncel
- [ ] Health check endpoint çalışıyor (`/health`)
- [ ] Test edilmiş local ortam
- [ ] Backup alındı

---

## 🚨 Önemli Notlar

1. **Environment Variables**: Tüm platformlarda manuel eklemeniz gerekir
2. **CSV Dosyaları**: Repo'da olmalı (Git'e commit edilmiş)
3. **Domain**: Her platformda custom domain ekleyebilirsiniz
4. **SSL**: Tüm platformlar otomatik SSL sağlar
5. **Backup**: Geçiş öncesi mutlaka backup alın

---

## 📞 Destek

- **Railway**: https://railway.app/docs
- **Fly.io**: https://fly.io/docs
- **DigitalOcean**: https://docs.digitalocean.com/products/app-platform/
- **PythonAnywhere**: https://help.pythonanywhere.com/

