"""`configs/sources.yaml`: hangi şirketlerin ilanları toplanacak ve hangileri tutulacak."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from flyhire.ingest.ats import FetchedJob, Provider

DEFAULT_CONFIG = Path("configs/sources.yaml")


class Filters(BaseModel):
    title_include: list[str] = Field(default_factory=list)
    title_exclude: list[str] = Field(default_factory=list)
    location_include: list[str] = Field(default_factory=list)
    location_exclude: list[str] = Field(default_factory=list)

    @field_validator("title_include", "title_exclude", "location_include", "location_exclude")
    @classmethod
    def patterns_must_compile(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            re.compile(pattern)
        return patterns

    def accepts(self, job: FetchedJob) -> bool:
        """Boş liste o filtreyi kapatır. Lokasyonu olmayan ilan lokasyon filtresinden geçmez.

        Lokasyon `;` veya `|` ile ayrılmış birden fazla yer içerebilir ("Remote, Canada; Remote,
        Poland"). İlan, en az bir yer `location_include`'a uyup `location_exclude`'a uymuyorsa
        tutulur.
        """
        if self.title_include and not _hit(self.title_include, job.title):
            return False
        if _hit(self.title_exclude, job.title):
            return False
        if not self.location_include:
            return True
        places = re.split(r"[;|]", job.location or "")
        return any(
            _hit(self.location_include, place) and not _hit(self.location_exclude, place)
            for place in places
        )


def _hit(patterns: list[str], value: str | None) -> bool:
    return any(re.search(p, value or "", re.IGNORECASE) for p in patterns)


class Board(BaseModel):
    provider: Provider
    slug: str
    company: str


class SourcesConfig(BaseModel):
    filters: Filters = Field(default_factory=Filters)
    boards: list[Board]

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "SourcesConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
