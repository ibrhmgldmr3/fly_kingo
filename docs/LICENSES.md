# Lisanslar ve atıflar

| Kaynak | Lisans | Projede kullanım | Not |
|---|---|---|---|
| [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (commit `91bdd1e`) | MIT | `Completeness_783.csv`, `Connectivity_783.parquet`, model parametreleri | MIT olduğu için kod ve veri doğrudan kullanılabilir. Lisans metni ve atıf korunmalı. |
| [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) (commit `a83b277`) | Repoda lisans dosyası yok | `neuron_annotations.tsv` (Supplemental file 1) | Schlegel et al., Nature 634 (2024) makalesinin ek dosyası. Makaleye atıf yapılmalı. |
| FlyWire connectome v783 | FlyWire kullanım koşulları | Bağlantı verisinin kaynağı | Dorkenwald et al., Nature 634 (2024). README'de atıf ve FlyWire teşekkür metni yer almalı. |
| [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) | GPL-2.0 | **Kullanılmıyor** | Kod kopyalanmaz. Kendi PyTorch LIF döngümüz yazılır; repo MIT kalır. |

Veri dosyaları repoya girmez, `flyhire data download` ile sabit commit'lerden iner ve SHA-256 ile doğrulanır.
