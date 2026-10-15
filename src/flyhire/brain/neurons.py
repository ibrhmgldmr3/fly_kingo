"""Tat kanalları ve okunan nöronlar (FlyWire kök ID'leri).

Listeler Shiu et al. 2024 `figures.ipynb` dosyasından alınmıştır ve FlyWire v630 kök ID'leridir;
kaynağa sadık kalmak için aynen tutulur. v783'te proofreading ile bazı kök ID'ler değişti:

- şeker: 21 ID'nin 20'si v783'te aynen var; 720575940620900446 → 720575940639259967
- acı:   21 ID'nin 20'si v783'te aynen var; 720575940618600651 → 720575940619072513
- su:    18/18 aynen var
- MN9:   Shiu'nun okuduğu MN9 (720575940660219265) aynen var. Karşı taraftaki MN9'un v630 ID'si
         (720575940645521262) yok; v783 karşılığı aynı hücre tipindeki (CB0701) 720575940618238523.

GRN eşlemeleri CAVE ile bulundu (`scripts/map_missing_grns.py`): iki durumda da eski kök tamamen tek
bir v783 köküne geçmiş (supervoxel örtüşmesi 1,0) ve başka parçalarla birleşmiş; bölünme yok. Yeni
kökler başka hiçbir listede değil ve v783 tipleri kanalla uyumlu (LB3c şeker, LB1c acı).
`TASTE_CHANNELS` bu eşlemeyle v783 ID'lerini içerir.

Eksik ID'ler sessizce atlanmaz: `resolve` eksikleri döndürür, testler sayıları sabitler.

Not: v783 annotasyonları (Schlegel et al. 2024) bu listeleri Shiu'dan farklı tipliyor: "şeker"
listesinde LB3d (yüksek tuz) ve LB4b, "su" listesinde LB3c (şeker) tipleri var. Projede makalede
doğrulanmış davranışı korumak için Shiu listeleri kullanılır; tiplemeye dayalı alternatif kanal
tanımları Faz 5 kontrollerinde karşılaştırılabilir.
"""

from collections.abc import Iterable

SUGAR_GRN: tuple[int, ...] = (
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
)  # fmt: skip

BITTER_GRN: tuple[int, ...] = (
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093,
    720575940626287336, 720575940618600651, 720575940627692048, 720575940630195909,
    720575940646212996, 720575940610483162, 720575940645743412, 720575940627578156,
    720575940622298631, 720575940621008895, 720575940629146711, 720575940610259370,
    720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
)  # fmt: skip

WATER_GRN: tuple[int, ...] = (
    720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053,
    720575940622902535, 720575940616177458, 720575940660292225, 720575940622486922,
    720575940613786774, 720575940629852866, 720575940625861168, 720575940613996959,
    720575940617857694, 720575940644965399, 720575940625203504, 720575940630553415,
    720575940635172191, 720575940634796536,
)  # fmt: skip

MN9 = 720575940660219265  # Shiu'nun okuduğu MN9: tat girdileriyle aynı taraf
MN9_CONTRA = 720575940618238523  # karşı taraftaki MN9 (v783)

# v630 kök ID → v783 kök ID (CAVE `get_latest_roots`, v783 zaman damgası)
V783_REPLACEMENTS: dict[int, int] = {
    720575940620900446: 720575940639259967,  # şeker GRN, LB3c
    720575940618600651: 720575940619072513,  # acı GRN, LB1c
}


def to_v783(ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(V783_REPLACEMENTS.get(i, i) for i in ids)


TASTE_CHANNELS: dict[str, tuple[int, ...]] = {
    "sugar": to_v783(SUGAR_GRN),
    "water": to_v783(WATER_GRN),
    "bitter": to_v783(BITTER_GRN),
}

READOUT: dict[str, int] = {"mn9": MN9, "mn9_contra": MN9_CONTRA}


def resolve(ids: Iterable[int], available: Iterable[int]) -> tuple[list[int], list[int]]:
    """ID'leri modelde bulunan ve bulunmayan olarak ayırır (sıra korunur)."""
    available = set(available)
    ids = list(ids)
    return [i for i in ids if i in available], [i for i in ids if i not in available]
