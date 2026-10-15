# Hortum Uzatma Kariyer Danışmanlığı: İlerleme Planı ve Teknoloji Yığını

## Context

`C:\projeler\00__keyfi\fly_kingo` klasöründe henüz kod yok, yalnızca iki tasarım dokümanı var:

- `meyve-sinegi-beyni-yol-haritasi.pdf` (28 Eylül 2026): Genel araştırma. Yol Haritası A (sinek beyni modülü), Yol Haritası B (senin katmanın), serbestlik bütçesi L0–L4, kontrol deneyleri, repo listesi.
- `hortum-uzatma-kariyer-danismanligi.pdf` (14 Ekim 2026): Somut ürün. İş ilanlarını LLM ile skorluyor, skorları şeker/su/acı tat nöronlarına spike olarak veriyor, Shiu LIF modeliyle (FlyWire v783, ağırlıklar sabit) MN9 hortum motor nöronunun ateşleme oranını ölçüyor, tek bir eşikle "başvur / geç" kararı çıkarıyor. Son onay insanda.

**Alınan kararlar:**
- Kapsam: Hortum Uzatma odaklı. A1–A3 sinek beyni modülü altyapı olarak kurulacak. Koku-ödül/FlyGym işi Faz 6+'ya kalıyor.
- Donanım: yerelde CPU, ağır simülasyonlar Colab/Kaggle'da.
- LLM: sağlayıcıdan bağımsız, `.env` üzerinden seçilecek.

**Hedef:** Önce 4 fazlık bir MVP (beyin → ilan hattı → kodlayıcı/eşik → panel), ardından kontroller, öğrenme ve yayın. Proje bilimsel açıdan dürüst kalmalı: sinek beyni (yeşil) ile senin katmanın (turuncu) arasındaki sınır kodda da görünmeli.

---

## Teknoloji yığını

| Katman | Seçim | Gerekçe / not |
|---|---|---|
| Dil, ortam | **Python 3.12**, **conda ortamı `fly_kingo`** (`environment.yml`), bağımlılıklar `pyproject.toml`'da, `pip install -e ".[dev]"` ile kurulum. ruff, pytest, pre-commit | Brian2, PyTorch (CPU) ve Streamlit 3.12'yi destekliyor. Conda yalnızca Python'u ve ağır ikili paketleri (numpy/scipy) sağlar, gerisi pip ile kurulur. Colab'da (Python 3.13) aynı `pyproject.toml` kullanılır. |
| Beyin, referans | **Brian2** + Shiu modelinin parametreleri (`philshiu/Drosophila_brain_model`) | Doğruluk referansı. Tam beyinle yalnızca Colab'da çalıştırılır. |
| Beyin, hızlı | **Kendi PyTorch LIF döngümüz** (`torch.sparse` CSR, batch = deneme sayısı, CPU/CUDA otomatik) | Eon'un `fly-brain` reposu GPL-2.0. Kod kopyalanmayacak, yalnızca okunarak örnek alınacak. Böylece repo MIT/Apache kalabilir. Shiu reposunun lisansı Faz 0'da kontrol edilecek. |
| Connectome verisi | FlyWire v783 (connections + `flywire_annotations` + `drosophila_neurotransmitters`), yükleme `pandas`/`pyarrow` ile | Veri repoya girmeyecek. `flyhire data download` (sabit commit + SHA-256). CC-BY atfı README'de yer alacak. |
| Yol analizi | `connectome_interpreter` (opsiyonel) | GRN → MN9 yollarını belgelemek için. |
| İlan toplama | `google-api-python-client` (Gmail, sadece `gmail.readonly`), `httpx`, `feedparser`, Greenhouse/Lever açık JSON uçları, `trafilatura`, Playwright (yalnızca gerekirse) | Doküman §4.1'deki ToS-dostu sıra korunacak. |
| LLM | **Instructor** (`instructor.from_provider("<sağlayıcı>/<model>")`) + **Pydantic v2** şemaları | Tek arayüzle Anthropic / OpenAI / Ollama çalışır. Model `.env`'deki `LLM_MODEL` değişkeninden gelir, şemaya uymayan çıktıda otomatik yeniden deneme yapılır. |
| Embedding | `sentence-transformers` + `intfloat/multilingual-e5-base` (CPU'da makul), alternatif `BAAI/bge-m3` | TR+EN karışık ilanlar için. |
| Veritabanı | **PostgreSQL 16 + pgvector** (Docker Compose, `pgvector/pgvector:pg16`), **SQLAlchemy 2** + **Alembic** | Windows'ta en temiz kurulum Docker Desktop ile. |
| Zamanlama | **APScheduler** (günlük toplama + gece toplu simülasyon) | Ek servis gerektirmez. |
| Analiz / değerlendirme | NumPy, pandas, scikit-learn (ROC, lojistik regresyon), **MLflow** (yerel, `mlruns/`) | Hesap açmadan deney takibi. W&B istenirse sonra eklenir. |
| Konfigürasyon | `pydantic-settings` (`.env`) + YAML (`configs/encoder.yaml`, `brain.yaml`, `sources.yaml`) | Hydra bu boyuttaki bir proje için fazla. Dondurulmuş kodlayıcı ayarları YAML'da versiyonlanır. |
| Arayüz | **Streamlit** + **Plotly** (spike raster, karar yüzeyi, sinek karnesi) | |
| Opsiyonel | FlyGym (Faz 7 animasyon), mantar cisimciği için kendi küçük alt ağ simülatörümüz (Faz 6) | |
| CI | GitHub Actions: ruff + pytest (hızlı testler), ağır beyin testleri `@pytest.mark.slow` ile işaretli | |

### CPU kısıtı için mimari karar: karar yüzeyi önbelleği
Beynin girdisi yalnızca üç frekanstan oluşuyor (şeker, su, acı; açlık şeker kazancını değiştiriyor). Bu yüzden MN9 oranı **3 boyutlu bir fonksiyon**. Plan şöyle:
1. Faz 1'de Colab GPU'da yoğun bir ızgara hesaplanır: örneğin 0–200 Hz, her eksende 21 nokta, nokta başına 5 tekrar. Sonuç `data/surface_v783.npz` olarak kaydedilir.
2. Günlük kullanımda (CPU) ilan kararı bu yüzeyden interpolasyonla okunur. Gece toplu işte ilanların bir alt kümesi **gerçek simülasyonla** çalıştırılır ve interpolasyon hatası raporlanır.
3. Panelde "tam simülasyon" / "yüzeyden" etiketi gösterilir. README'de bu bir varsayım olarak yazılır.
4. Spike raster için, panelde açılan ilan isteğe bağlı olarak CPU'da kısa bir tam simülasyonla çalıştırılır (süre Faz 1'de ölçülecek).

Faz 6'da plastisite durum ekleyeceği için bu önbellek oraya uygulanmaz. Mantar cisimciği alt ağı küçük, CPU'da çalışır.

---

## Repo yapısı (hedef)

```
fly_kingo/
├─ environment.yml · pyproject.toml · .env.example · docker-compose.yml · README.md · LICENSE
├─ docs/                      # mevcut iki PDF buraya taşınır, ARCHITECTURE.md, ASSUMPTIONS.md
├─ configs/                   # brain.yaml, encoder.yaml (dondurulmuş), sources.yaml
├─ data/                      # .gitignore: flywire/, surface_*.npz, cache/
├─ notebooks/colab_surface.ipynb
├─ src/flyhire/
│  ├─ brain/       # YEŞİL. connectome.py, neurons.py (GRN/MN9 ID'leri), lif_brian2.py,
│  │               #  lif_torch.py, api.py → simulate(inputs_hz, duration, n_trials) -> rates, surface.py
│  ├─ ingest/      # gmail.py, rss.py, greenhouse.py, lever.py, clean.py, dedupe.py
│  ├─ extract/     # schema.py (JobFeatures), llm.py (Instructor), embed.py
│  ├─ encoder/     # TURUNCU. taste.py (skor→Hz), hunger.py, redflags.py (sözlük)
│  ├─ readout/     # TURUNCU. threshold.py (tek parametre, ROC)
│  ├─ db/          # models.py, session.py, alembic/
│  ├─ eval/        # baselines.py (kural, LR, embedding), controls.py (shuffled), metrics.py
│  ├─ learning/    # Faz 6: mushroom body
│  ├─ jobs/        # scheduler.py
│  └─ app/         # Streamlit sayfaları
└─ tests/          # test_sugar_mn9.py, test_bitter_suppresses.py, test_encoder.py, ...
```

**Sınır kuralı (kodda zorunlu):** `encoder/`, `readout/` ve `app/` beyne yalnızca `brain.api.simulate(...)` üzerinden erişir. Ağırlık tensörleri `requires_grad=False` ve salt-okunur olacak. Bunu doğrulayan bir test yazılacak: simülasyondan önceki ve sonraki ağırlık hash'i aynı kalmalı.

---

## İlerleme planı (yarı zamanlı)

### Faz 0: Kurulum · 2–3 gün
- Conda ortamı: `conda create -n fly_kingo python=3.12 -y` (karşılığı `environment.yml`: `name: fly_kingo`, `python=3.12`, `pip`, numpy, scipy). Ardından `conda activate fly_kingo` ve `pip install -e ".[dev]"`. PyTorch CPU sürümü `--index-url https://download.pytorch.org/whl/cpu` ile kurulur.
- `pyproject.toml` (src layout, `flyhire` paketi, `[project.scripts] flyhire = ...` CLI), klasör yapısı, ruff/pytest/pre-commit, git init, GitHub reposu (`flyhire` / `proboscis-extension-recruiting`).
- `docker-compose.yml` ile Postgres+pgvector (port **5433**; 5432 makinedeki yerel PostgreSQL servisinde). Alembic ilk migration (jobs, features, decisions, applications, labels).
- `flyhire data download`: FlyWire v783 (Shiu reposundaki işaretli bağlantılar) + annotations. ✅
- Shiu reposunun ve FlyWire verisinin lisansını kontrol et, `docs/LICENSES.md`'ye yaz.
- **Bitti tanımı:** `conda activate fly_kingo` sonrası `python --version` 3.12, `pytest` yeşil, `docker compose up` ile DB ayakta, veri iniyor.

### Faz 1: Sinek beynini ayağa kaldır · ✅ tamamlandı (14 Ekim 2026)
Ayrıntılı sonuçlar: [BRAIN_VALIDATION.md](BRAIN_VALIDATION.md).
- ✅ `neurons.py`: Shiu listeleri (v630). v783'te aynen bulunmayan iki GRN CAVE ile eşlendi, kanallar 21/18/21; MN9 aynen var. Karşı taraf MN9'un v783 karşılığı (CB0701) bulundu. Eşlenen iki GRN MN9'u ortalama ~2 Hz, eşik bölgesinde 10–15 Hz'e kadar değiştiriyor (doğrulama raporu §1).
- ✅ `lif_brian2.py`: Brian2 referansı (Colab gerekmedi; yerelde cython ile ~48 s/deneme).
- ✅ `lif_torch.py`: PyTorch modeli. Deterministik girdide Brian2 ile **spike spike aynı**; tam beyinde korelasyon 0,999, MN9 89,7 / 89,8 Hz. Brian2'nin refrakter sürede sinaptik girdiyi yok sayma kuralı bulundu ve uygulandı.
- ✅ Kalıcı testler: `test_sugar_activates_mn9`, `test_bitter_suppresses_sugar`, `test_water_activates_mn9`, `test_weights_unchanged_after_simulation` (`pytest -m slow`), ayrıca hızlı paket.
- ✅ CPU performansı: 1,45 s/deneme (1 s sanal süre). İlan başına 5 denemelik tam simülasyon ≈ 7 s.
- ✅ `api.py`: `FlyBrain.simulate()` / `simulate_batch()`; diğer katmanların beyne tek giriş noktası.
- ✅ `surface.py`: şeker×su×acı yüzeyi (11×6×11 nokta, 5 deneme). İlk sürüm CPU'da 81 dakikada, 21/18/21 sürümü kısmen Colab T4'te hesaplandı. İnterpolasyon hatası (20/18/20 yüzeyi) ortalama 2,1 Hz (en büyük 6,0 Hz). `flyhire brain check-surface` interpolasyon hatasını ölçer, `flyhire brain heatmap` README görselini çizer.
- **Plandan sapma:** PyTorch modeli CPU'da yeterince hızlı, ama uzun CPU/GPU koşuları dizüstü bilgisayarı kapattığı için ağır hesaplar `notebooks/colab_brain.ipynb` ile Colab'da yapılıyor.

### Faz 2: İlan boru hattı · 1,5–2 hafta
- Gmail iş alarmı okuyucusu (LinkedIn, Kariyer.net, Indeed alarmları), link ayıklama.
- 1–2 RSS/API kaynağı (Remote OK, Greenhouse/Lever board'ları). `trafilatura` ile temiz metin. Hash ile tekrar ayıklama.
- `JobFeatures` Pydantic şeması (doküman §4.2), Instructor ile çıkarım, 20 ilanda elle kontrol.
  - ✅ Şema ve çıkarım: `src/flyhire/extract/`, `flyhire extract ilan.txt`. Kırmızı bayrak alıntıları ilanda birebir geçmek zorunda (geçmezse LLM'den düzeltmesi istenir). Gerçek LLM testleri `pytest -m llm` (API anahtarı ister).
  - ⏳ 20 ilanda elle kontrol.
- CV ve ilan embedding'leri pgvector'a, kosinüs benzerliği. (Repoya **sahte CV**. Gerçek CV yalnızca yerelde.)
- APScheduler günlük toplama.
- **Bitti tanımı:** tablo her gün otomatik doluyor, özellikler ve benzerlik skorları hesaplanmış.

### Faz 3: Tat kodlayıcı ve kalibrasyon · 1 hafta
- `redflags.py`: doküman §8'deki sözlük + LLM'in yakaladığı bayraklar.
- `taste.py`: §4.3 eşlemesi → kanal başına 0–1 skor → 0–200 Hz. `hunger.py`: son başvurudan geçen gün → şeker kazancı (varsayım olarak işaretli).
- 50–100 ilanı elle etiketle (`labels` tablosu, basit bir Streamlit etiketleme sayfası). Train/test ayrımı sabit seed ile.
- Katkı oranlarını bir kez ayarla → `configs/encoder.yaml`'da dondur. Eşiği ROC ile seç (L1 seviyesi).
- **Bitti tanımı:** dondurulmuş config, eşik, test setinde ilk doğruluk/kesinlik değerleri MLflow'da.

### Faz 4: Onay paneli (MVP) · 1,5–2 hafta
- Liste (MN9'a göre sıralı, rozetler: "Hortum uzatıldı" / "Kafasını çevirdi" / "Kararsız, kanatlarını temizliyor"), detay sayfası (özellikler, tat skorları, bayrak alıntıları, spike raster), "Başvur / Geç / Sinek yanılıyor" butonları.
- Ön yazı taslağı: yalnızca `my_experience.md` kaynağından (RAG). Kaynak dışına çıkmama talimatı verilir, taslağı her zaman kullanıcı düzenler. Gönderme otomasyonu **yok**.
- Takip durumları: bulundu → onaylandı → gönderildi → yanıt.
- `flyhire demo` modu: sahte CV + örnek ilanlarla uçtan uca çalışma.
- 1 hafta gerçek kullanım.
- **Bitti tanımı:** günlük kullanılabilir MVP, demo modu, "Sinek yanılıyor" etiketleri birikiyor.

**MVP toplamı: yaklaşık 6–8 hafta.**

### Faz 5: Kontroller ve değerlendirme · 1 hafta
- Baselines: düz kural (şeker+su−acı>eşik), lojistik regresyon, yalnızca embedding, karıştırılmış connectome (aynı derece dağılımı, aynı NT işaretleri). Karıştırılmış yüzey de Colab'da hesaplanır.
- Metrikler: accuracy/precision/recall, tekrar tutarlılığı, interpolasyon hatası, sinek ile kuralın ayrıştığı ilanların tek tek incelenmesi.
- README'ye sonuç tablosu ve dürüst sonuç paragrafı ("Sinek iyi bir İK uzmanı mı?").

### Faz 6: Mantar cisimciği ile öğrenme (opsiyonel) · 3–4 hafta
- PN → KC → MBON / DAN alt ağı. Şirket adından deterministik ORN deseni (koku).
- KC→MBON'a dopamin kapılı LTD. Önce sentetik testler (aynı şirketten üst üste ret).
- Kaçınma MBON'larını readout'a ikinci girdi olarak ekle, açılıp kapanabilir anahtar, açık/kapalı farkını ölç. Bağlantının bir tasarım tercihi olduğu README'de belirtilir.

### Faz 7: Görselleştirme ve yayın · 1–2 hafta
- Sinek animasyonu (FlyGym veya hazır Three.js/Blender klibi), demo videosu, README (doküman §9 yapısı), LinkedIn paylaşımı, blog yazısı, projenin `awesome-fly` listesine eklenmesi.

**Tüm fazlar: yaklaşık 12–16 hafta.**

---

## Riskler ve önlemler

| Risk | Önlem |
|---|---|
| GRN/MN9 ID'leri veya Shiu sonucunun tekrar üretilememesi | Faz 1 ilk sırada. Shiu reposundaki ID listeleri ve parametreler birebir kullanılır, Brian2 referansı Colab'da çalıştırılır. |
| CPU'da simülasyonun yavaş olması | Karar yüzeyi önbelleği, gece toplu iş, yalnızca açılan ilan için tam simülasyon. |
| GPL bulaşması | Eon kodu kopyalanmaz, kendi LIF döngümüz yazılır. Lisanslar `docs/LICENSES.md`'de. |
| Platform ToS ihlali | Gmail alarmları ve açık API'ler kullanılır, giriş gerektiren sayfa kazınmaz, otomatik başvuru yapılmaz. |
| Kişisel veri sızıntısı | CV, e-posta ve token'lar `.env` ile `data/private/` altında (gitignore). Repoda sahte CV. Yayında şirket adları anonimleştirilir. |
| Kodlayıcının kararı ele geçirmesi | Kodlayıcı eğitilmez, bir kez ayarlanıp dondurulur. Tek eğitilen parametre eşik (L1). Faz 5 kontrolleri bunu ölçer. |

---

## Doğrulama

Tüm komutlar `conda activate fly_kingo` sonrası çalıştırılır (ya da `conda run -n fly_kingo <komut>`).
- `pytest`: hızlı birim testleri (kodlayıcı, şema, tekrar ayıklama, eşik, ağırlık dondurma).
- `pytest -m slow` (Colab veya uzun CPU koşusu): `test_sugar_mn9` (şeker uyarımında MN9 oranı belirlenen eşiğin üstünde) ve `test_bitter_suppresses` (şeker+acı < şeker).
- `docker compose up -d` → `flyhire ingest --once` → `flyhire score --once` → `streamlit run src/flyhire/app/main.py`: uçtan uca akış.
- `flyhire demo`: sahte veriyle panelin açılması, rozetler ve spike raster.
- Faz 5'te MLflow'daki karşılaştırma tablosunun README'ye aktarılması.

## Uygulamaya geçince ilk adımlar
1. `fly_kingo` conda ortamını (Python 3.12) oluştur, ardından Faz 0 iskeletini kur (pyproject, klasörler, docker-compose, veri indirme scripti), PDF'leri `docs/` altına taşı.
2. Faz 1 için `neurons.py` ve Colab notebook'unu hazırla.
