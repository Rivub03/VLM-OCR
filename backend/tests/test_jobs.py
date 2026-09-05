import asyncio

from app.cache import ResultCache
from app.jobs import JobManager


def test_running_job_can_be_cancelled() -> None:
    async def scenario() -> None:
        manager = JobManager(ResultCache(60))
        started = asyncio.Event()

        async def work():
            started.set()
            await asyncio.sleep(60)

        response = manager.start("job-1", work)
        assert response.status == "queued"
        await started.wait()
        assert manager.cancel("job-1").status == "cancelled"
        await manager.shutdown()

    asyncio.run(scenario())
