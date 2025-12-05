# Web TÜFE - Günlük Tüketici Fiyat Endeksi

Web TÜFE, Türkiye'nin günlük tüketici fiyat endeksini (TÜFE) takip etmenizi sağlayan bir platformdur.

## Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
```

### İletişim Formu için SMTP Ayarları

İletişim formunun çalışması için Gmail SMTP ayarlarını yapmanız gerekiyor:

#### 1. Gmail App Password Oluşturma

1. [Google Hesabınıza](https://myaccount.google.com/) gidin
2. **Güvenlik** sekmesine tıklayın
3. **2 Adımlı Doğrulama**'yı aktifleştirin (henüz aktif değilse)
4. **Uygulama Şifreleri**'ne gidin
5. "Diğer (Özel ad)" seçeneğini seçin ve "Web TÜFE" yazın
6. **Oluştur**'a tıklayın
7. Gösterilen 16 haneli şifreyi kopyalayın (boşlukları kaldırın)

#### 2. Environment Variables Ayarlama

**Production (Render, Heroku, vb.):**

Environment variable'lar hosting platformunuzda ayarlanmalıdır. Uygulama önce sistem environment variable'larını kontrol eder.

**Local Development (Opsiyonel - .env dosyası):**

Local'de test etmek için proje klasörünüzde `.env` dosyası oluşturabilirsiniz:

1. Proje klasörünüzde `.env` adında bir dosya oluşturun
2. Aşağıdaki içeriği ekleyin:

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=borakaya8@gmail.com
SMTP_PASSWORD=your_app_password_here
ADMIN_API_KEY=your_admin_api_key_here
```

**Not:** `.env` dosyası sadece local development içindir. Production'da (Render'da) environment variable'lar otomatik olarak kullanılır ve `.env` dosyası göz ardı edilir.

**Alternatif: Manuel Environment Variables (Geçici)**

**Windows (PowerShell):**
```powershell
$env:SMTP_SERVER="smtp.gmail.com"
$env:SMTP_PORT="587"
$env:SMTP_USERNAME="borakaya8@gmail.com"
$env:SMTP_PASSWORD="abcdefghijklmnop"  # Yukarıda oluşturduğunuz app password
```

**Windows (Command Prompt):**
```cmd
set SMTP_SERVER=smtp.gmail.com
set SMTP_PORT=587
set SMTP_USERNAME=borakaya8@gmail.com
set SMTP_PASSWORD=abcdefghijklmnop
```

**Linux/Mac:**
```bash
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="borakaya8@gmail.com"
export SMTP_PASSWORD="abcdefghijklmnop"
```

#### 3. Production Ortamında (Render, Heroku, vb.)

Hosting platformunuzun environment variables bölümünden aşağıdaki değişkenleri ekleyin:

- `SMTP_SERVER`: `smtp.gmail.com`
- `SMTP_PORT`: `587`
- `SMTP_USERNAME`: `borakaya8@gmail.com`
- `SMTP_PASSWORD`: (Gmail App Password)

### Development Modu

SMTP şifresi ayarlanmadığında, form mesajları console'a yazdırılır ve başarılı mesajı döner (test için).

## Çalıştırma

```bash
python app.py
```

Uygulama `http://localhost:5000` adresinde çalışacaktır.

## Özellikler

- Günlük TÜFE takibi
- Ana gruplar ve harcama grupları analizi
- İnteraktif grafikler
- PDF bültenler
- Metodoloji sayfası
- İletişim formu (SMTP ile email gönderimi)

## Güvenlik Notları

- SMTP şifrenizi asla GitHub'a yüklemeyin
- Environment variables kullanın
- `.env` dosyası kullanıyorsanız `.gitignore`'a ekleyin
- Production ortamında güçlü şifreler kullanın

## Lisans

Bu proje kişisel kullanım içindir.
