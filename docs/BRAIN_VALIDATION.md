# Sinek beyni modülü: doğrulama raporu (Faz 1)

Tarih: 14 Ekim 2026 · Connectome: FlyWire v783 (138.639 nöron, 15.091.983 bağlantı) ·
Model: Shiu et al. 2024 LIF, kendi PyTorch uygulamamız (`src/flyhire/brain/lif_torch.py`)

## 1. Nöron kimlikleri (v630 → v783)

Shiu'nun listeleri FlyWire v630 kök ID'leri. v783'te:

| Kanal | Shiu listesi | v783'te bulunan | Eksik |
|---|---|---|---|
| Şeker GRN | 21 | 20 | `720575940620900446` |
| Su GRN | 18 | 18 | - |
| Acı GRN | 21 | 20 | `720575940618600651` |
| MN9 (okunan) | `720575940660219265` | aynen var | - |
| MN9 (karşı taraf) | `720575940645521262` | yok | v783 karşılığı aynı hücre tipinden (CB0701): `720575940618238523` |

Eksik iki GRN yerine yenisi eklenmedi; kanallar 20/18/20 nöronla çalışıyor. Bu sayılar
`tests/test_brain_full.py::test_channel_coverage` ile sabitlendi.

**Tipleme farkı:** v783 annotasyonları (Schlegel et al. 2024) Shiu'nun "şeker" listesinde LB3d
(yüksek tuz) ve LB4b, "su" listesinde LB3c (şeker) tipleri gösteriyor. Makalede doğrulanmış davranışı
korumak için Shiu listeleri kullanıldı; tiplemeye dayalı kanallar Faz 5'te kontrol olarak denenebilir.

## 2. PyTorch modelinin Brian2 ile eşdeğerliği

Brian2 referansı: `src/flyhire/brain/lif_brian2.py` (Shiu `model.py` temelli, MIT).

| Test | Sonuç |
|---|---|
| Küçük rastgele ağ (300 nöron), deterministik girdi, 1 s | 18.919 spike, **tüm nöronlarda birebir aynı sayı** |
| Aynı ağ, Poisson girdi, 400 / 150 deneme | uyarılmayan nöronların toplam oranı 639,4 ± 2,4 / 636,9 ± 3,8 Hz |
| **Tam beyin**, şeker 200 Hz, torch 20 / Brian2 5 deneme | aktif 406 nöron, korelasyon **0,9992**, toplam oran 16.170 / 16.191 Hz, MN9 **89,7 / 89,8 Hz** |

Deterministik eşdeğerlik testi hızlı test paketinde (`test_matches_brian2_exactly_with_deterministic_input`).

### Bulunan ince nokta: refrakter sürede sinaptik girdi
İlk sürümde aşağı akış nöronları %5–12 fazla ateşliyordu. Sebep: Brian2, `(unless refractory)`
işaretli değişkenlere **koşullu yazma** uygular. Refrakter (veya o adımda spike atmış) bir nörona gelen
`g += w` sinaptik girdisi geldiği anda **yok sayılır**; yalnızca türev dondurulmaz. Bu davranış küçük
bir Brian2 deneyiyle ve Brian2 2.10 kaynak koduyla (`NeuronGroup`, `set_conditional_write`)
doğrulandı. PyTorch modeli aynı kuralı uygular.

## 3. Biyolojik sonuçların tekrar üretilmesi

Tam beyin, 1 s sanal süre, 10 deneme (CPU):

| Girdi | MN9 (Hz) | Beklenen (Shiu 2024) |
|---|---|---|
| Girdi yok | 0,0 | sessiz |
| Şeker 100 Hz | 57,6 ± 4,6 | ateşler |
| Şeker 200 Hz | 92,1 ± 6,9 | ateşler, frekansla artar |
| Şeker 100 + acı 100 Hz | **2,6 ± 1,7** | acı şeker yanıtını baskılar |
| Şeker 100 + acı 200 Hz | 0,0 | |
| Su 200 Hz | 31,7 ± 4,3 | ateşler |
| Acı 200 Hz | 0,0 | ateşlemez |

Kalıcı testler: `pytest -m slow` (yaklaşık 1 dakika).

**Kodlayıcı için not (Faz 3):** acı çok güçlü bir baskılayıcı. Şeker ile aynı frekanstaki acı, MN9'u
neredeyse tamamen susturuyor. Acı kanalının ölçeği bu yüzden şekerinkinden çok daha küçük seçilmeli;
karar yüzeyinde acı ekseni düşük frekanslarda sık örneklendi.

## 4. Performans (CPU, 16 çekirdek)

| | Süre |
|---|---|
| Connectome yükleme | 0,6 s |
| PyTorch, 1 s sanal süre | **1,45 s / deneme** (batch 10–30) |
| Brian2 (cython), 1 s sanal süre | ~48 s / deneme (ağ kurulumu dahil) |

İlk PyTorch sürümü 3,8 s/deneme idi. Refrakter nöronları tüm dizi üzerinde maske yerine yalnızca son
spike'lar üzerinden izlemek 2,6× hızlandırdı (`v_rst = v_0` ve refrakterde `g = 0` olduğu için bu
nöronlar zaten dinlenme noktasında).

Sonuç: karar yüzeyi (726 nokta × 5 deneme) **CPU'da 81 dakikada** hesaplandı; Colab
şart değil. İlan başına tam simülasyon (5 deneme) yaklaşık 7 saniye.

## 5. Karar yüzeyi

`data/surface_v783.npz`: şeker 0–200 Hz (11 nokta) × su 0–200 Hz (6) × acı 0–200 Hz (11, düşük
frekanslarda sık), nokta başına 5 deneme, 1 s. CPU'da 81 dakikada hesaplandı.

![Şeker × acı → MN9](figures/decision_surface.png)

Yüzeyden çıkan, basit bir toplamla açıklanamayacak davranışlar:

- **Şeker eşiği:** 40 Hz'e kadar MN9 neredeyse sessiz (2 Hz), 60 Hz'de 32 Hz'e sıçrıyor.
- **Su tek başına zayıf, şekeri güçlendiriyor:** su 120 Hz'e kadar MN9'u ateşlemiyor (200 Hz'de 32 Hz),
  ama şeker 60 Hz'in yanında su 80 Hz yanıtı 32'den 62 Hz'e çıkarıyor.
- **Acı baskılaması şekere bağlı:** düşük şekerde az acı yetiyor, yüksek şekerde çok daha fazlası gerekiyor.
  Su 120 Hz'de şeker eşiği ~20 Hz'e iniyor ve acıya tolerans artıyor
  ([görsel](figures/decision_surface_water120.png)).

Bunlar Faz 5'te "sinek ile basit kural nerede ayrışıyor?" sorusunun aday cevapları.

### İnterpolasyon hatası

`flyhire brain check-surface --points 24`: ızgara dışındaki 24 rastgele noktada yüzey (doğrusal
interpolasyon) ile gerçek simülasyon (5 deneme) karşılaştırıldı.

| | Değer |
|---|---|
| Ortalama mutlak hata | **2,1 Hz** |
| En büyük hata | 6,0 Hz |
| Simülasyonun kendi denemeler arası sapması | ± 0–7 Hz |

Hata, simülasyonun kendi gürültüsüyle aynı büyüklükte. Günlük kararlar yüzeyden okunabilir; eşiğe
yakın ilanlar (panelde "kararsız") için tam simülasyon çalıştırılmalı.
