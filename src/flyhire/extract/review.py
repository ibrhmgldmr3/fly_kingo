"""Özellik çıkarımının elle kontrolü (Faz 2: "20 ilanda elle kontrol").

`export_review` rastgele seçilen ilanların çıkarılan alanlarını bir CSV'ye yazar (alan başına bir
satır). Kontrol eden kişi ilanı linkinden okur ve `dogru` sütununa `e` (doğru) veya `h` (yanlış)
yazar; isterse `not` sütununa doğrusunu not eder. `score_review` alan bazında doğruluk oranını
verir.

CSV Excel'in Türkçe ayarlarıyla doğrudan açılsın diye `;` ile ayrılır ve UTF-8 BOM ile yazılır.
"""

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from flyhire.db.models import Job, JobFeatures

COLUMNS = ["job_id", "sirket", "baslik", "url", "alan", "deger", "dogru", "not"]
FIELDS = [
    "required",
    "preferred",
    "experience",
    "seniority",
    "work_mode",
    "locations",
    "salary",
    "flexible_hours",
    "red_flags",
]
# features-v2 ve öncesinin alan adları (eski kontrol dosyaları da puanlanabilsin)
OLD_FIELDS = {
    "required_skills": "required",
    "preferred_skills": "preferred",
    "location": "locations",
}


def _value(data: dict, name: str) -> str:
    if name in {"required", "preferred"}:
        items = data.get(name) or []
        return "; ".join(" veya ".join(item["any_of"]) for item in items) or "(yok)"
    if name == "flexible_hours":
        quote = data.get("flexible_hours_quote")
        return f"True: “{quote}”" if data.get("flexible_hours") else "False"
    if name == "experience":
        years = data.get("min_years_experience")
        quote = data.get("experience_quote")
        return "(yazmıyor)" if years is None else f"{years:g} yıl: “{quote}”"
    if name == "salary":
        salary = data.get("salary")
        if not salary:
            return "(yok)"
        low, high = salary.get("min"), salary.get("max")
        amount = " - ".join(f"{v:,.0f}" for v in (low, high) if v is not None)
        return f"{amount} {salary.get('currency') or ''} / {salary.get('period') or '?'}".strip()
    if name == "red_flags":
        flags = data.get("red_flags") or []
        return " | ".join(f"{f['kind']}: “{f['quote']}”" for f in flags) or "(yok)"
    value = data.get(name)
    if isinstance(value, list):
        return ", ".join(value) or "(yok)"
    return "(yok)" if value is None else str(value)


def export_review(session: Session, path: Path, n: int = 20, seed: int = 0) -> list[int]:
    """`n` ilanı rastgele (sabit seed) seçip kontrol CSV'sini yazar; seçilen id'leri döndürür."""
    rows = session.execute(select(Job, JobFeatures).join(JobFeatures).order_by(Job.id)).all()
    chosen = sorted(random.Random(seed).sample(rows, min(n, len(rows))), key=lambda r: r[0].id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(COLUMNS)
        for job, features in chosen:
            for name in FIELDS:
                value = _value(features.data, name)
                writer.writerow([job.id, job.company, job.title, job.url, name, value, "", ""])
    return [job.id for job, _ in chosen]


@dataclass
class ReviewScore:
    correct: Counter
    wrong: Counter
    unmarked: int

    def lines(self) -> list[str]:
        out = []
        for name in FIELDS:
            total = self.correct[name] + self.wrong[name]
            if total:
                rate = self.correct[name] / total
                out.append(f"{name:<17} {self.correct[name]:>3}/{total:<3} %{rate * 100:5.1f}")
        total_ok, total = (
            sum(self.correct.values()),
            sum(self.correct.values()) + sum(self.wrong.values()),
        )
        if total:
            out.append(f"{'TOPLAM':<17} {total_ok:>3}/{total:<3} %{total_ok / total * 100:5.1f}")
        if self.unmarked:
            out.append(f"işaretlenmemiş satır: {self.unmarked}")
        return out


def score_review(path: Path) -> ReviewScore:
    correct, wrong, unmarked = Counter(), Counter(), 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = ";" if sample.count(";") >= sample.count(",") else ","
        for row in csv.DictReader(f, delimiter=delimiter):
            mark = (row.get("dogru") or "").strip().casefold()
            field = OLD_FIELDS.get(row["alan"], row["alan"])
            if mark in {"e", "evet", "d", "doğru", "dogru", "1", "yes"}:
                correct[field] += 1
            elif mark in {"h", "hayır", "hayir", "y", "yanlış", "yanlis", "0", "n", "no"}:
                wrong[field] += 1
            else:
                unmarked += 1
    return ReviewScore(correct, wrong, unmarked)
