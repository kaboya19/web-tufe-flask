# TÜFE Analiz Web Uygulaması

Bu Flask tabanlı web uygulaması, Google Sheets'ten TÜFE (Tüketici Fiyat Endeksi) verilerini çekerek modern bir arayüzde görselleştirir.

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Google Sheets API için kimlik bilgilerini ayarlayın:
   - Google Cloud Console'dan bir proje oluşturun
   - Google Sheets API'yi etkinleştirin
   - Servis hesabı oluşturun ve JSON formatında kimlik bilgilerini indirin
   - İndirdiğiniz JSON dosyasını `credentials.json` olarak projenin kök dizinine kaydedin
   - Google Sheets dokümanınızı servis hesabı e-postası ile paylaşın

3. Uygulamayı çalıştırın:
```bash
python app.py
```

4. Tarayıcınızda `http://localhost:5000` adresine gidin

## Özellikler

- Google Sheets'ten otomatik veri çekme
- Modern ve duyarlı tasarım
- Yatay bar grafiği görselleştirmesi
- TÜFE değerinin kırmızı renkte vurgulanması
- Değerlere göre sıralanmış gösterim 