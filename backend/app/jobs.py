import asyncio
from dataclasses import dataclass

from .cache import ResultCache
from .schemas import JobResponse, OCRResult


@dataclass
class Job:
    status: str = "queued"
    result: OCRResult | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = None


class JobManager:
    """Small in-process job registry.

    It deliberately does not queue source files or persist work.  Cancellation
    propagates into the HTTP request to the inference server and releases the
    application concurrency slot as soon as the client connection is closed.
    """

    def __init__(self, results: ResultCache):
        self.jobs: dict[str, Job] = {}
        self.results = results

    def start(self, job_id: str, work) -> JobResponse:
        job = Job()
        self.jobs[job_id] = job

        async def run() -> None:
            job.status = "running"
            try:
                job.result = await work()
                job.status = "completed"
                self.results.put(job_id, job.result)
            except asyncio.CancelledError:
                job.status = "cancelled"
                raise
            except Exception as exc:  # Errors are returned by the status route.
                job.status = "failed"
                job.error = str(exc) or "OCR processing failed."

        job.task = asyncio.create_task(run(), name=f"ocr-job-{job_id}")
        return self.response(job_id, job)

    def response(self, job_id: str, job: Job | None = None) -> JobResponse:
        job = job or self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return JobResponse(job_id=job_id, status=job.status, result=job.result, error=job.error)

    def cancel(self, job_id: str) -> JobResponse:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status in {"queued", "running"}:
            job.status = "cancelled"
            if job.task:
                job.task.cancel()
        return self.response(job_id, job)

    async def shutdown(self) -> None:
        pending = [job.task for job in self.jobs.values() if job.task and not job.task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
