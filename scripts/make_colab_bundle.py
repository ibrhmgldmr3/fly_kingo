"""Colab için kod paketi: git'in gördüğü dosyalar (commit edilmemişler dahil), veri hariç.

`notebooks/colab_brain.ipynb` bu zip'i yükleyerek GitHub'a push etmeden çalışabilir.

    python scripts/make_colab_bundle.py
"""

import subprocess
import zipfile
from pathlib import Path

OUT = Path("fly_kingo_colab.zip")


def main() -> None:
    # takip edilen + .gitignore'da olmayan yeni dosyalar; data/* gibi büyük veriler hariç
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode()
    paths = sorted({Path(p) for p in listed.split("\0") if p} - {OUT})
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            if path.is_file():
                z.write(path, path.as_posix())
    print(f"{OUT}: {len(paths)} dosya, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
