"""HTTP client for job-svc (GET /api/jobs/{id}) with a short TTL cache so scoring does not inflate ViewCount."""
import time
import uuid

import httpx

from app.errors import BadRequest, JobNotFound, JobServiceUnavailable


class JobClient:
    def __init__(self, base_url: str, timeout: float, ttl: int):
        self.base = base_url.rstrip("/")
        self.ttl = ttl
        self._client = httpx.AsyncClient(timeout=timeout)
        self._cache: dict[str, tuple[float, dict]] = {}

    async def get(self, job_id: str) -> dict:
        try:
            uuid.UUID(job_id)
        except (ValueError, AttributeError):
            raise BadRequest("job_id must be a UUID.") from None
        hit = self._cache.get(job_id)
        if hit and hit[0] > time.time():
            return hit[1]
        if not self.base:
            raise JobServiceUnavailable("JOB_SERVICE_URL is not configured.")
        try:
            r = await self._client.get(f"{self.base}/api/jobs/{job_id}")
        except httpx.HTTPError as exc:
            raise JobServiceUnavailable("Could not reach the job service.") from exc
        if r.status_code == 404:
            raise JobNotFound("No job with that id.")
        if r.status_code != 200:
            raise JobServiceUnavailable(f"Job service returned HTTP {r.status_code}.")
        job = r.json()
        self._cache[job_id] = (time.time() + self.ttl, job)
        return job

    async def close(self) -> None:
        await self._client.aclose()
