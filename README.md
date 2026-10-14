# Hortum Uzatma Kariyer Danışmanlığı (FlyHire)

> İş başvurularıma 139.255 nöronlu bir meyve sineği karar veriyor.

İş ilanlarını bir LLM ile skorlayıp sineğin tat nöronlarına (şeker, su, acı) spike olarak veren,
kararı ise FlyWire connectome'u üzerinde çalışan Shiu LIF modelinde **MN9 hortum motor nöronunun**
ateşleme oranından çıkaran bir öneri sistemi. Son onay ve gönderme her zaman insanda.

**Durum:** Faz 1 tamamlandı: sinek beyni çalışıyor ve Shiu et al. 2024'ün sonuçlarını tekrar üretiyor
([doğrulama raporu](docs/BRAIN_VALIDATION.md)). Yol haritası: [docs/ROADMAP.md](docs/ROADMAP.md).

| Girdi (1 s, 10 deneme) | MN9 |
|---|---|
| Şeker 200 Hz | 92 Hz, hortum uzatılır |
| Şeker 100 Hz + acı 100 Hz | 3 Hz, acı şekeri bastırır |
| Su 200 Hz | 32 Hz |

![Sineğin karar yüzeyi](docs/figures/decision_surface.png)

```bash
flyhire brain simulate --sugar 100 --bitter 20   # tek girdi, ~15 s
flyhire brain surface                            # karar yüzeyi, ~80 dk, devam ettirilebilir
flyhire brain heatmap                            # docs/figures/decision_surface.png
```

| Katman | Kim karar veriyor? |
|---|---|
| İlan toplama, LLM özellik çıkarımı, tat kodlayıcı, eşik | Senin katmanın (turuncu) |
| Şeker/su/acı → MN9 | **Sinek beyni** (yeşil, ağırlıklar sabit) |
| Başvuru | İnsan |

## Kurulum

```bash
conda env create -f environment.yml
conda activate fly_kingo
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[brain,dev]"
cp .env.example .env
docker compose up -d
alembic upgrade head
flyhire data download
flyhire info
pytest            # hızlı testler (~15 s), Brian2 ile birebir eşdeğerlik dahil
pytest -m slow    # tam beyin biyolojik testleri (~1 dk)
```

## Dokümanlar

- [docs/hortum-uzatma-kariyer-danismanligi.pdf](docs/hortum-uzatma-kariyer-danismanligi.pdf): proje tasarımı
- [docs/meyve-sinegi-beyni-yol-haritasi.pdf](docs/meyve-sinegi-beyni-yol-haritasi.pdf): arka plan araştırması
- [docs/LICENSES.md](docs/LICENSES.md): veri ve kod lisansları

*Decisions made by 139,255 neurons. No fruit flies were harmed.*
