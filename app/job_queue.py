"""Simple async job queue with in-memory storage."""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class JobType(str, Enum):
    """Type of job."""

    GENERATE = "generate"
    GENERATE_SOUNDTRACK = "generate_soundtrack"
    SEPARATE = "separate"
    GENERATE_ACESTEP = "generate_acestep"


class JobStatus(str, Enum):
    """Job status enumeration."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Job representation."""

    id: str
    job_type: JobType
    status: JobStatus
    params: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result_urls: list[str] | None = None
    error: str | None = None
    progress: float = 0.0
    progress_detail: dict[str, Any] | None = None


class JobQueue:
    """In-memory async job queue."""

    def __init__(self, max_concurrent: int = 1) -> None:
        """Initialize job queue.

        Args:
            max_concurrent: Maximum concurrent jobs (default 1 for GPU)
        """
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._max_concurrent = max_concurrent
        self._workers: list[asyncio.Task] = []
        self._handlers: dict[JobType, Callable[[Job], Coroutine[Any, Any, list[str]]]] = {}
        self._running = False
        self._lock = asyncio.Lock()

        # Metrics
        self._total_jobs = 0
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._generation_times: list[float] = []
        self._soundtrack_times: list[float] = []
        self._separation_times: list[float] = []
        self._acestep_times: list[float] = []

    def register_handler(
        self,
        job_type: JobType,
        handler: Callable[[Job], Coroutine[Any, Any, list[str]]],
    ) -> None:
        """Register a handler for a job type.

        Args:
            job_type: Type of job
            handler: Async function that processes the job and returns result URLs
        """
        self._handlers[job_type] = handler

    async def start(self) -> None:
        """Start the job queue workers."""
        if self._running:
            return
        self._running = True
        for i in range(self._max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        logger.info(f"Started {self._max_concurrent} job queue workers")

    async def stop(self) -> None:
        """Stop the job queue workers."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Stopped job queue workers")

    async def enqueue(self, job_type: JobType, params: dict[str, Any]) -> Job:
        """Add a job to the queue.

        Args:
            job_type: Type of job
            params: Job parameters

        Returns:
            Created job
        """
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            params=params,
        )

        async with self._lock:
            self._jobs[job_id] = job
            self._total_jobs += 1

        await self._queue.put(job_id)
        logger.info(f"Enqueued job {job_id} of type {job_type}")
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job if found, None otherwise
        """
        return self._jobs.get(job_id)

    def get_metrics(self) -> dict[str, Any]:
        """Get queue metrics.

        Returns:
            Metrics dictionary
        """
        in_progress = sum(1 for j in self._jobs.values() if j.status == JobStatus.PROCESSING)
        return {
            "jobs_total": self._total_jobs,
            "jobs_completed": self._completed_jobs,
            "jobs_failed": self._failed_jobs,
            "jobs_in_progress": in_progress,
            "avg_generation_time_seconds": (
                sum(self._generation_times) / len(self._generation_times)
                if self._generation_times
                else None
            ),
            "avg_soundtrack_time_seconds": (
                sum(self._soundtrack_times) / len(self._soundtrack_times)
                if self._soundtrack_times
                else None
            ),
            "avg_separation_time_seconds": (
                sum(self._separation_times) / len(self._separation_times)
                if self._separation_times
                else None
            ),
            "avg_acestep_time_seconds": (
                sum(self._acestep_times) / len(self._acestep_times)
                if self._acestep_times
                else None
            ),
        }

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes jobs.

        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} started")
        while self._running:
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            job = self._jobs.get(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found")
                continue

            handler = self._handlers.get(job.job_type)
            if not handler:
                logger.error(f"No handler for job type {job.job_type}")
                job.status = JobStatus.FAILED
                job.error = f"No handler for job type {job.job_type}"
                self._failed_jobs += 1
                continue

            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            logger.info(f"Worker {worker_id} processing job {job_id}")

            try:
                result_urls = await handler(job)
                job.status = JobStatus.COMPLETED
                job.result_urls = result_urls
                job.completed_at = time.time()

                duration = job.completed_at - job.started_at
                if job.job_type == JobType.GENERATE:
                    self._generation_times.append(duration)
                elif job.job_type == JobType.GENERATE_SOUNDTRACK:
                    self._soundtrack_times.append(duration)
                elif job.job_type == JobType.SEPARATE:
                    self._separation_times.append(duration)
                elif job.job_type == JobType.GENERATE_ACESTEP:
                    self._acestep_times.append(duration)

                self._completed_jobs += 1
                logger.info(f"Job {job_id} completed in {duration:.2f}s")

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
                self._failed_jobs += 1
                logger.exception(f"Job {job_id} failed: {e}")

    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up jobs older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of jobs cleaned up
        """
        cutoff = time.time() - (max_age_hours * 3600)
        to_remove = []

        async with self._lock:
            for job_id, job in self._jobs.items():
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    if job.completed_at and job.completed_at < cutoff:
                        to_remove.append(job_id)

            for job_id in to_remove:
                del self._jobs[job_id]

        logger.info(f"Cleaned up {len(to_remove)} old jobs")
        return len(to_remove)


# Global job queue instance
job_queue = JobQueue(max_concurrent=1)
