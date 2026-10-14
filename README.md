# Hortum Uzatma Kariyer Danışmanlığı (FlyHire)

> İş başvurularıma 139.255 nöronlu bir meyve sineği karar veriyor.

İş ilanlarını bir LLM ile skorlayıp sineğin tat nöronlarına (şeker, su, acı) spike olarak veren,
kararı ise FlyWire connectome'u üzerinde çalışan Shiu LIF modelinde **MN9 hortum motor nöronunun**
ateşleme oranından çıkaran bir öneri sistemi. Son onay ve gönderme her zaman insanda.

**Durum:** Faz 0 (kurulum). Yol haritası: [docs/ROADMAP.md](docs/ROADMAP.md).

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
pytest
```

## Dokümanlar

- [docs/hortum-uzatma-kariyer-danismanligi.pdf](docs/hortum-uzatma-kariyer-danismanligi.pdf): proje tasarımı
- [docs/meyve-sinegi-beyni-yol-haritasi.pdf](docs/meyve-sinegi-beyni-yol-haritasi.pdf): arka plan araştırması
- [docs/LICENSES.md](docs/LICENSES.md): veri ve kod lisansları

*Decisions made by 139,255 neurons. No fruit flies were harmed.*
