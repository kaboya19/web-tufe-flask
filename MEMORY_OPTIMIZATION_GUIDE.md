# Render Memory ve CPU Optimizasyon Rehberi

## 🚨 Sorun: RAM ve CPU Limitlerine Ulaşma

Render'da RAM ve CPU kullanımı limitlere ulaşıyorsa, şu optimizasyonları yapın:

## ✅ Yapılan Optimizasyonlar

### 1. Worker Sayısı Azaltıldı
- **Önceki**: 4 worker
- **Şimdi**: 2 worker
- **Etkisi**: RAM kullanımı ~%50 azalır

### 2. Cache Threshold Azaltıldı
- **Önceki**: 2000 item
- **Şimdi**: 500 item
- **Etkisi**: Memory kullanımı azalır

### 3. Preload Eklendi
- `--preload` flag'i eklendi
- **Etkisi**: Memory paylaşımı, daha az RAM kullanımı

### 4. Max Requests Azaltıldı
- **Önceki**: 1000 request
- **Şimdi**: 500 request
- **Etkisi**: Memory leak önleme

## 🔧 Ek Optimizasyon Önerileri

### 1. Render Planınızı Kontrol Edin

**Free Plan Limitleri:**
- RAM: 512 MB
- CPU: 0.5 vCPU
- **Sorun**: Bu çok az!

**Çözüm:**
- **Starter Plan** ($7/ay): 512 MB RAM, 0.5 vCPU (yeterli değil)
- **Professional Plan** ($25/ay): 2 GB RAM, 1 vCPU (önerilen)

### 2. Memory Kullanımını İzleyin

Render Dashboard → Metrics → Memory
- **%80'in altında** olmalı
- **%90+** → Sorun var, plan yükseltin

### 3. DataFrame Copy() İşlemlerini Azaltın

Kodunuzda şu satırları bulun ve optimize edin:

```python
# ❌ Kötü: Her seferinde copy
df = df_raw.copy()

# ✅ İyi: Sadece gerektiğinde copy
df = df_raw  # View kullan, copy yapma
# veya
df = df_raw.copy() if need_modification else df_raw
```

### 4. Cache Timeout'u Azaltın

Eğer hala memory sorunu varsa:

```python
# app.py'de
cache_config = {
    'CACHE_DEFAULT_TIMEOUT': 300,  # 5 dakikaya düşür (600'den)
    'CACHE_THRESHOLD': 300  # 300'e düşür (500'den)
}
```

### 5. Gereksiz DataFrame İşlemlerini Kaldırın

```python
# ❌ Kötü: Her seferinde yeni DataFrame
df = pd.DataFrame(data)
df = df.sort_values()
df = df.reset_index()

# ✅ İyi: Tek seferde
df = pd.DataFrame(data).sort_values().reset_index()
```

### 6. Memory-Efficient CSV Okuma

```python
# ❌ Kötü: Tüm dosyayı okur
df = pd.read_csv('large_file.csv')

# ✅ İyi: Sadece ihtiyacınız olan sütunları oku
df = pd.read_csv('large_file.csv', usecols=['col1', 'col2'])
```

### 7. Plotly Grafiklerini Optimize Edin

```python
# ❌ Kötü: Büyük grafikler
fig = go.Figure(data=[...], layout={...})

# ✅ İyi: Sadece gerekli veriler
fig = go.Figure(data=[...], layout={...})
fig.update_layout(height=400)  # Yüksekliği sınırla
```

## 📊 Memory Kullanımını İzleme

### Render Dashboard
1. **Metrics** sekmesine gidin
2. **Memory Usage** grafiğini izleyin
3. **CPU Usage** grafiğini izleyin

### Kritik Eşikler
- **Memory > %80**: Uyarı
- **Memory > %90**: Kritik
- **CPU > %80**: Uyarı
- **CPU > %95**: Kritik

## 🚀 Hızlı Çözümler

### Çözüm 1: Worker Sayısını Azalt (YAPILDI)
```yaml
# render.yaml
startCommand: gunicorn app:app --workers 2 --threads 2
```

### Çözüm 2: Cache Threshold Azalt (YAPILDI)
```python
# app.py
'CACHE_THRESHOLD': 500  # 2000'den 500'e
```

### Çözüm 3: Render Planını Yükselt
- Dashboard → Settings → Plan
- **Professional Plan** seçin ($25/ay)
- 2 GB RAM, 1 vCPU

### Çözüm 4: Timeout Azalt
```yaml
# render.yaml
--timeout 60  # 120'den 60'a
```

## ⚠️ Acil Durum: Memory Limit Aşılırsa

### 1. Worker Sayısını 1'e Düşürün
```yaml
startCommand: gunicorn app:app --workers 1 --threads 4
```

### 2. Cache'i Tamamen Kapatın (Geçici)
```python
# app.py'de cache'i devre dışı bırakın
cache_config = {
    'CACHE_TYPE': 'NullCache'  # Cache yok
}
```

### 3. Render Planını Yükseltin
- **En hızlı çözüm**: Professional Plan ($25/ay)

## 📈 Beklenen İyileştirmeler

### Worker 4 → 2
- **RAM**: ~%50 azalma
- **CPU**: ~%30 azalma
- **Performans**: ~%10-20 yavaşlama (kabul edilebilir)

### Cache Threshold 2000 → 500
- **RAM**: ~%25 azalma
- **Cache Hit Rate**: Biraz düşebilir (ama sorun değil)

### Preload Flag
- **RAM**: ~%10 azalma
- **Startup**: Biraz daha hızlı

## 🎯 Öncelik Sırası

1. ✅ **Worker sayısını azalt** (YAPILDI)
2. ✅ **Cache threshold azalt** (YAPILDI)
3. ⚠️ **Render planını yükselt** (Önerilen)
4. ⚠️ **DataFrame copy() optimize et** (Kod değişikliği gerekir)
5. ⚠️ **Memory kullanımını izle** (Sürekli)

## 💡 Sonuç

**Şu an yapılanlar:**
- Worker: 4 → 2 (RAM %50 azalır)
- Cache threshold: 2000 → 500 (RAM %25 azalır)
- Preload: Eklendi (RAM %10 azalır)

**Toplam RAM tasarrufu: ~%60-70**

**Eğer hala sorun varsa:**
1. Render planını yükseltin (Professional Plan)
2. Worker sayısını 1'e düşürün
3. Cache'i tamamen kapatın (geçici)

## 📞 Destek

Render Support:
- Email: support@render.com
- Docs: https://render.com/docs

