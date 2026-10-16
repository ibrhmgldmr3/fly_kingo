"""Greenhouse, Lever ve Ashby'nin herkese açık ilan uç noktaları.

Bu uç noktalar şirketlerin kariyer sayfalarını beslemek için açıkça yayınlanır; giriş veya anahtar
gerektirmez. Her şirket için tek bir istek yapılır (ilanların tamamı tek yanıtta gelir).

- Greenhouse: https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true
- Lever:      https://api.lever.co/v0/postings/{slug}?mode=json
- Ashby:      https://api.ashbyhq.com/posting-api/job-board/{slug}
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from flyhire.extract.schema import JobPosting
from flyhire.ingest.clean import content_hash, html_to_text

Provider = Literal["greenhouse", "lever", "ashby"]

USER_AGENT = "flyhire/0.1 (+https://github.com/ibrhmgldmr3/fly_kingo)"


class FetchedJob(BaseModel):
    """Kaynaktan bağımsız ilan kaydı; `db.models.Job` alanlarıyla aynı."""

    source: str  # "lever:trendyol"
    external_id: str
    url: str
    company: str
    title: str
    location: str | None
    text: str
    posted_at: datetime | None

    @property
    def content_hash(self) -> str:
        return content_hash(self.company, self.title, self.text)

    def to_posting(self) -> JobPosting:
        return JobPosting(
            text=self.text,
            title=self.title,
            company=self.company,
            location=self.location,
            url=self.url,
        )


def _join(*parts: str | None) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _location(place: str | None, workplace: str | None) -> str | None:
    """Kaynakların ayrı alanda verdiği çalışma şeklini lokasyona ekler (LLM metinde görsün)."""
    workplace = (workplace or "").strip().lower().replace("_", " ")
    if workplace and workplace not in (place or "").lower():
        return f"{place} ({workplace})" if place else workplace
    return place


def parse_greenhouse(data: dict[str, Any], slug: str, company: str) -> list[FetchedJob]:
    jobs = []
    for job in data.get("jobs", []):
        posted = job.get("first_published") or job.get("updated_at")
        jobs.append(
            FetchedJob(
                source=f"greenhouse:{slug}",
                external_id=str(job["id"]),
                url=job["absolute_url"],
                company=job.get("company_name") or company,
                title=job["title"].strip(),
                location=(job.get("location") or {}).get("name"),
                text=html_to_text(job.get("content") or "", escaped=True),
                posted_at=datetime.fromisoformat(posted) if posted else None,
            )
        )
    return jobs


def _plain_or_html(job: dict[str, Any], field: str) -> str:
    """Lever bazen yalnızca HTML alanını doldurur (`descriptionPlain` boş, `description` dolu)."""
    return job.get(f"{field}Plain") or html_to_text(job.get(field) or "")


def parse_lever(data: list[dict[str, Any]], slug: str, company: str) -> list[FetchedJob]:
    jobs = []
    for job in data:
        categories = job.get("categories") or {}
        place = categories.get("location") or ", ".join(categories.get("allLocations") or [])
        if job.get("country"):
            place = f"{place}, {job['country']}" if place else job["country"]
        sections = [
            _join(item.get("text"), html_to_text(item.get("content") or ""))
            for item in job.get("lists") or []
        ]
        created = job.get("createdAt")
        jobs.append(
            FetchedJob(
                source=f"lever:{slug}",
                external_id=job["id"],
                url=job["hostedUrl"],
                company=company,
                title=job["text"].strip(),
                location=_location(place or None, job.get("workplaceType")),
                text=_join(
                    _plain_or_html(job, "description"),
                    *sections,
                    _plain_or_html(job, "additional"),
                ),
                posted_at=datetime.fromtimestamp(created / 1000, UTC) if created else None,
            )
        )
    return jobs


def parse_ashby(data: dict[str, Any], slug: str, company: str) -> list[FetchedJob]:
    jobs = []
    for job in data.get("jobs", []):
        if job.get("isListed") is False:
            continue
        secondary = [s.get("location") for s in job.get("secondaryLocations") or []]
        places = [job.get("location"), *secondary]
        place = "; ".join(p for p in places if p) or None
        published = job.get("publishedAt")
        jobs.append(
            FetchedJob(
                source=f"ashby:{slug}",
                external_id=job["id"],
                url=job["jobUrl"],
                company=company,
                title=job["title"].strip(),
                location=_location(place, job.get("workplaceType")),
                text=job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml") or ""),
                posted_at=datetime.fromisoformat(published) if published else None,
            )
        )
    return jobs


_ENDPOINTS: dict[Provider, tuple[str, Callable[..., list[FetchedJob]]]] = {
    "greenhouse": (
        "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true",
        parse_greenhouse,
    ),
    "lever": ("https://api.lever.co/v0/postings/{slug}?mode=json", parse_lever),
    "ashby": ("https://api.ashbyhq.com/posting-api/job-board/{slug}", parse_ashby),
}


def fetch_board(
    provider: Provider, slug: str, company: str, client: httpx.Client | None = None
) -> list[FetchedJob]:
    url, parse = _ENDPOINTS[provider]
    own_client = client is None
    client = client or httpx.Client(
        timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )
    try:
        response = client.get(url.format(slug=slug))
        response.raise_for_status()
        return parse(response.json(), slug, company)
    finally:
        if own_client:
            client.close()
