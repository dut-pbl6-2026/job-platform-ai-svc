"""HTTP client for search-svc semantic job search (GET /api/search/jobs/semantic). ai-svc never queries Elasticsearch."""
import httpx

from app.errors import SearchUnavailable

MAX_HITS = 3


class SearchClient:
    def __init__(self, base_url: str, timeout: float):
        self.base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def find_jobs(self, query: str) -> list[dict]:
        """Up to MAX_HITS jobs as {id, title, company}, most similar first. Raises SearchUnavailable on any upstream failure."""
        if not self.base:
            raise SearchUnavailable("SEARCH_SERVICE_URL is not configured.")
        try:
            r = await self._client.get(f"{self.base}/api/search/jobs/semantic", params={"q": query, "size": MAX_HITS})
        except httpx.HTTPError as exc:
            raise SearchUnavailable("Could not reach the search service.") from exc
        if r.status_code != 200:
            raise SearchUnavailable(f"Search service returned HTTP {r.status_code}.")
        try:
            items = r.json().get("items")
        except (ValueError, AttributeError) as exc:
            raise SearchUnavailable("Search service returned an unreadable body.") from exc
        if not isinstance(items, list):
            raise SearchUnavailable("Search service returned no items list.")
        hits = []
        for item in items[:MAX_HITS]:
            job = item.get("job") if isinstance(item, dict) else None
            if not isinstance(job, dict) or not job.get("id"):
                continue
            hits.append({
                "id": str(job["id"]),
                "title": str(job.get("title") or ""),
                "company": str(job.get("companyName") or ""),
            })
        return hits

    async def close(self) -> None:
        await self._client.aclose()
