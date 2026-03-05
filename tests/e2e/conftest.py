"""Pytest fixtures for E2E tests."""

import io
import time
import wave

import pytest
import requests

from .config import API_KEY, API_URL, JOB_POLL_INTERVAL, JOB_TIMEOUT, REQUEST_TIMEOUT


@pytest.fixture
def api_url() -> str:
    """Return the API base URL."""
    return API_URL


@pytest.fixture
def api_key() -> str:
    """Return the API key."""
    if not API_KEY:
        pytest.skip("MUSICGEN_API_KEY not set and kubectl not available")
    return API_KEY


@pytest.fixture
def auth_headers(api_key: str) -> dict[str, str]:
    """Return headers with API key authentication."""
    return {"X-API-Key": api_key}


@pytest.fixture
def session() -> requests.Session:
    """Return a requests session with default timeout."""
    s = requests.Session()
    s.timeout = REQUEST_TIMEOUT
    return s


def wait_for_job(
    session: requests.Session,
    api_url: str,
    job_id: str,
    timeout: float = JOB_TIMEOUT,
    poll_interval: float = JOB_POLL_INTERVAL,
) -> dict:
    """
    Poll job status until completion or timeout.

    Returns the final job status dict.
    Raises TimeoutError if job doesn't complete in time.
    """
    start_time = time.time()
    last_status = None
    consecutive_errors = 0
    max_consecutive_errors = 5

    while time.time() - start_time < timeout:
        try:
            response = session.get(
                f"{api_url}/jobs/{job_id}",
                timeout=60,  # Longer timeout for status check during GPU processing
            )
            response.raise_for_status()
            status = response.json()
            last_status = status
            consecutive_errors = 0  # Reset on success

            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise RuntimeError(f"Job failed: {status.get('error', 'Unknown error')}")

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    f"Too many consecutive connection errors polling job {job_id}: {e}"
                )
            # Connection error during GPU processing is common, wait and retry
            time.sleep(poll_interval * 2)
            continue

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Job {job_id} did not complete within {timeout}s. Last status: {last_status}"
    )


@pytest.fixture
def wait_for_job_completion(session: requests.Session, api_url: str):
    """Fixture that returns a function to wait for job completion."""

    def _wait(job_id: str, timeout: float = JOB_TIMEOUT) -> dict:
        return wait_for_job(session, api_url, job_id, timeout=timeout)

    return _wait


def download_file(session: requests.Session, api_url: str, file_path: str) -> bytes:
    """Download a file from the API."""
    response = session.get(f"{api_url}{file_path}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


@pytest.fixture
def download(session: requests.Session, api_url: str):
    """Fixture that returns a function to download files."""

    def _download(file_path: str) -> bytes:
        return download_file(session, api_url, file_path)

    return _download


def get_wav_duration(wav_data: bytes) -> float:
    """Get duration of WAV file in seconds."""
    with wave.open(io.BytesIO(wav_data), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return frames / rate


def validate_wav(wav_data: bytes) -> dict:
    """Validate WAV file and return its properties."""
    with wave.open(io.BytesIO(wav_data), "rb") as wav:
        return {
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
            "framerate": wav.getframerate(),
            "frames": wav.getnframes(),
            "duration": wav.getnframes() / wav.getframerate(),
        }
