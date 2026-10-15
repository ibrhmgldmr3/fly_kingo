"""Shiu listelerindeki, v783'te bulunmayan GRN'lerin v783 karşılıklarını CAVE ile bulur.

Salt okunur sorgular yapar; neurons.py'yi değiştirmez. CAVE token'ı önceden kaydedilmiş olmalı.

    python scripts/map_missing_grns.py
"""

import argparse
from pathlib import Path

import pandas as pd
from caveclient import CAVEclient

from flyhire.brain import neurons

MISSING = {"sugar": 720575940620900446, "bitter": 720575940618600651}
DATA = Path("data/flywire")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-file",
        help="Kullanılacak CAVE token dosyası (varsayılan: caveclient'ın kendi seçimi)",
    )
    args = parser.parse_args()
    client = CAVEclient("flywire_fafb_public", auth_token_file=args.token_file)
    ts630 = client.materialize.get_timestamp(630)
    ts783 = client.materialize.get_timestamp(783)
    print(f"v630 zaman damgası: {ts630}\nv783 zaman damgası: {ts783}\n")

    model_ids = set(pd.read_csv(DATA / "Completeness_783.csv", index_col=0).index)
    ann = pd.read_csv(DATA / "neuron_annotations.tsv", sep="\t", low_memory=False)
    ann = ann.set_index("root_id")[["cell_type", "cell_sub_class", "side"]]
    all_listed = {i: name for name, ids in neurons.TASTE_CHANNELS.items() for i in ids}

    for channel, old in MISSING.items():
        print(f"=== {channel}: {old}")
        latest = [int(x) for x in client.chunkedgraph.get_latest_roots(old, timestamp=ts783)]
        print(f"get_latest_roots (v783): {latest}")
        suggested, overlap = client.chunkedgraph.suggest_latest_roots(
            old, timestamp=ts783, return_all=True, return_fraction_overlap=True
        )
        for root, frac in zip(suggested, overlap, strict=True):
            print(f"  suggest_latest_roots: {int(root)}  supervoxel örtüşmesi {frac:.3f}")

        for new in latest:
            past = [
                int(x)
                for x in client.chunkedgraph.get_past_ids([new], timestamp_past=ts630)[
                    "past_id_map"
                ][new]
            ]
            kind = []
            if len(latest) > 1:
                kind.append(f"bölünme ({len(latest)} parçaya)")
            if len(past) > 1:
                kind.append(f"birleşme (v630'da {len(past)} kökten)")
            if not kind:
                kind.append("yalnızca düzenleme (tek kök → tek kök)")
            others = [p for p in past if p != old]
            cell = ann.loc[new].to_dict() if new in ann.index else "annotasyon yok"
            print(f"  yeni kök {new}")
            print(f"    tür              : {', '.join(kind)}")
            print(f"    v630 geçmişi     : {past}")
            print(f"    diğer v630 kökleri listede mi: {[(p, all_listed.get(p)) for p in others]}")
            print(f"    zaten listede mi : {all_listed.get(new, 'hayır')}")
            print(f"    modelde (v783)   : {new in model_ids}")
            print(f"    annotasyon       : {cell}")
        print()


if __name__ == "__main__":
    main()
