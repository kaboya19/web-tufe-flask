# Railway Fiyatlandırma Açıklaması

## 🤔 Railway Nasıl Ücretlendirir?

### Temel Prensip
Railway **kaynak kullanımına** göre ücretlendirir, **trafiğe** göre değil.

## 📊 Ücretlendirme Modeli

### 1. CPU ve RAM Limitleri
Railway'da **ayarladığınız CPU ve RAM limitlerine** göre ücretlendirme yapılır:

```
2 CPU + 4 GB RAM ayarlarsanız:
- Sürekli bu kadar ücret ödersiniz
- Yoğunluk artarsa → Ücret DEĞİŞMEZ (aynı limitler)
- Ama performans düşebilir (limitler yetersiz kalırsa)
```

### 2. Saatlik Ücretlendirme
- **CPU**: $0.000231 per vCPU-hour
- **RAM**: $0.000463 per GB-hour
- **Sürekli çalışma**: 730 saat/ay

### 3. Ücretsiz Kredi
- **$5/ay** ücretsiz kredi
- Bu kredi CPU ve RAM ücretlerinden düşülür

## 🎯 Senaryolar

### Senaryo 1: Normal Kullanım
```
Ayarlar: 2 CPU + 4 GB RAM
Maliyet: ~$1.69/ay
Ücretsiz Kredi: $5/ay
Gerçek Ödeme: $0/ay ✅
```

### Senaryo 2: Yoğunluk Artarsa (Aynı Limitler)
```
Ayarlar: 2 CPU + 4 GB RAM (değişmedi)
Maliyet: ~$1.69/ay (DEĞİŞMEDİ)
Ücretsiz Kredi: $5/ay
Gerçek Ödeme: $0/ay ✅
Not: Performans düşebilir ama ücret aynı
```

### Senaryo 3: Yoğunluk Artarsa (Limitleri Artırırsanız)
```
Ayarlar: 4 CPU + 8 GB RAM (artırdınız)
Maliyet: ~$3.38/ay (2x arttı)
Ücretsiz Kredi: $5/ay
Gerçek Ödeme: $0/ay ✅ (hala ücretsiz kredi içinde)
```

### Senaryo 4: Çok Yoğunluk (Çok Yüksek Limitler)
```
Ayarlar: 8 CPU + 16 GB RAM (çok artırdınız)
Maliyet: ~$6.76/ay
Ücretsiz Kredi: $5/ay
Gerçek Ödeme: ~$1.76/ay ⚠️
```

## 💡 Önemli Noktalar

### ✅ Ücreti Etkileyenler
- **CPU limiti** (ne kadar CPU ayarladığınız)
- **RAM limiti** (ne kadar RAM ayarladığınız)
- **Çalışma süresi** (uygulama ne kadar süre çalışıyor)

### ❌ Ücreti Etkilemeyenler
- **Trafik miktarı** (kaç kullanıcı)
- **İstek sayısı** (kaç request)
- **Bandwidth** (ne kadar veri transferi)
- **Database kullanımı** (ayrı ücretlendirme varsa)

## 🎯 Sizin Durumunuz

### Render'da Sorun
- **512 MB RAM** (çok az)
- **0.5 vCPU** (çok az)
- **Yoğunluk** → Limitlere ulaşıyor → Site yavaşlıyor

### Railway'da Çözüm
- **4 GB RAM** ayarlayabilirsiniz (8x daha fazla)
- **2 CPU** ayarlayabilirsiniz (4x daha fazla)
- **Maliyet**: ~$1.69/ay
- **Ücretsiz kredi**: $5/ay
- **Gerçek ödeme**: $0/ay ✅

### Yoğunluk Artarsa
- **Aynı limitlerle kalırsanız**: Ücret değişmez ($0/ay)
- **Limitleri artırırsanız**: Ücret artar (ama hala $5 kredi içinde kalabilir)

## 📊 Karşılaştırma

| Durum | Render | Railway |
|-------|--------|---------|
| **Normal kullanım** | Ücretsiz (sınırlı) | ~$1.69/ay (ama $5 kredi var) |
| **Yoğunluk artarsa** | Limitlere ulaşır → Yavaşlar | Aynı limitler → Ücret değişmez |
| **Limitleri artırma** | Plan yükseltme gerekir ($25/ay) | Sadece limitleri artır ($3.38/ay) |
| **Gerçek maliyet** | $0/ay (ama yavaş) veya $25/ay | $0/ay (çoğu durumda) |

## 🎯 Sonuç

### Railway'da:
1. **2 CPU + 4 GB RAM** ayarlarsanız → ~$1.69/ay
2. **$5 ücretsiz kredi** var → Gerçek ödeme $0/ay
3. **Yoğunluk artarsa** → Ücret değişmez (aynı limitlerle)
4. **Performans sorunu olursa** → Limitleri artırabilirsiniz (ama ücret artar)

### Özet:
- **Yoğunluk = Ücret artışı DEĞİL**
- **Limit artışı = Ücret artışı EVET**
- **Çoğu durumda $5 kredi yeterli** → Ücretsiz kullanım

## 💰 Gerçek Örnekler

### Örnek 1: Küçük Site
```
CPU: 1 vCPU
RAM: 512 MB
Maliyet: ~$0.42/ay
Ücretsiz kredi: $5/ay
Ödeme: $0/ay ✅
```

### Örnek 2: Orta Site (Sizin Durumunuz)
```
CPU: 2 vCPU
RAM: 4 GB
Maliyet: ~$1.69/ay
Ücretsiz kredi: $5/ay
Ödeme: $0/ay ✅
```

### Örnek 3: Büyük Site
```
CPU: 4 vCPU
RAM: 8 GB
Maliyet: ~$3.38/ay
Ücretsiz kredi: $5/ay
Ödeme: $0/ay ✅
```

### Örnek 4: Çok Büyük Site
```
CPU: 8 vCPU
RAM: 16 GB
Maliyet: ~$6.76/ay
Ücretsiz kredi: $5/ay
Ödeme: ~$1.76/ay ⚠️
```

## 🎯 Sonuç

**Railway'da yoğunluk artarsa:**
- ✅ **Aynı limitlerle kalırsanız**: Ücret değişmez ($0/ay)
- ⚠️ **Limitleri artırırsanız**: Ücret artar (ama genelde $5 kredi içinde)

**Render'da yoğunluk artarsa:**
- ❌ **Limitlere ulaşır**: Site yavaşlar
- ❌ **Plan yükseltme**: $25/ay gerekir

**Railway daha esnek ve genelde daha ucuz!** 🚀

