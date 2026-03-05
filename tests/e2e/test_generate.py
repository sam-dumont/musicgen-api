"""E2E tests for music generation endpoint."""

import pytest
import requests

from .config import MEDIUM_DURATION, REQUEST_TIMEOUT, SHORT_DURATION
from .conftest import get_wav_duration, validate_wav


class TestMusicGeneration:
    """Full E2E tests for /generate endpoint."""

    @pytest.mark.timeout(300)
    def test_generate_short_music(
        self,
        session: requests.Session,
        api_url: str,
        auth_headers: dict,
        wait_for_job_completion,
        download,
    ):
        """Test generating a short piece of music (10 seconds)."""
        # Submit generation request
        response = session.post(
            f"{api_url}/generate",
            json={
                "prompt": "upbeat electronic dance music with synth leads",
                "duration": SHORT_DURATION,
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Wait for completion
        result = wait_for_job_completion(job_id, timeout=180)
        assert result["status"] == "completed"
        assert result["result_urls"] is not None
        assert len(result["result_urls"]) == 1

        # Download and validate result
        wav_data = download(result["result_urls"][0])
        props = validate_wav(wav_data)

        # Check duration is approximately correct (allow some tolerance)
        assert props["duration"] >= SHORT_DURATION - 1
        assert props["duration"] <= SHORT_DURATION + 2
        assert props["channels"] in [1, 2]  # Mono or stereo
        assert props["framerate"] == 32000  # MusicGen output sample rate

    @pytest.mark.timeout(300)
    def test_generate_with_mood(
        self,
        session: requests.Session,
        api_url: str,
        auth_headers: dict,
        wait_for_job_completion,
        download,
    ):
        """Test generating music with a mood modifier."""
        response = session.post(
            f"{api_url}/generate",
            json={
                "prompt": "orchestral film score",
                "duration": SHORT_DURATION,
                "mood": "epic and dramatic",
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        result = wait_for_job_completion(job_id, timeout=180)
        assert result["status"] == "completed"

        wav_data = download(result["result_urls"][0])
        props = validate_wav(wav_data)
        assert props["duration"] >= SHORT_DURATION - 1

    @pytest.mark.timeout(600)
    def test_generate_medium_duration(
        self,
        session: requests.Session,
        api_url: str,
        auth_headers: dict,
        wait_for_job_completion,
        download,
    ):
        """Test generating longer music (30 seconds, requires sliding window)."""
        response = session.post(
            f"{api_url}/generate",
            json={
                "prompt": "relaxing jazz piano with soft drums",
                "duration": MEDIUM_DURATION,
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        result = wait_for_job_completion(job_id, timeout=300)
        assert result["status"] == "completed"

        wav_data = download(result["result_urls"][0])
        props = validate_wav(wav_data)

        # 30s uses sliding window generation
        assert props["duration"] >= MEDIUM_DURATION - 2
        assert props["duration"] <= MEDIUM_DURATION + 3

    def test_job_progress_tracking(
        self,
        session: requests.Session,
        api_url: str,
        auth_headers: dict,
    ):
        """Test that job progress can be tracked."""
        import time

        # Submit a job
        response = session.post(
            f"{api_url}/generate",
            json={
                "prompt": "ambient soundscape",
                "duration": SHORT_DURATION,
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Poll for status a few times
        statuses_seen = set()
        for _ in range(60):  # Max 60 polls
            status_response = session.get(
                f"{api_url}/jobs/{job_id}",
                timeout=REQUEST_TIMEOUT,
            )
            assert status_response.status_code == 200
            data = status_response.json()

            statuses_seen.add(data["status"])
            assert "progress" in data

            if data["status"] == "completed":
                break
            elif data["status"] == "failed":
                pytest.fail(f"Job failed: {data.get('error')}")

            time.sleep(3)

        # Should have seen at least queued or processing before completed
        assert "completed" in statuses_seen or "processing" in statuses_seen

    def test_nonexistent_job_returns_404(
        self,
        session: requests.Session,
        api_url: str,
    ):
        """Test that querying a non-existent job returns 404."""
        response = session.get(
            f"{api_url}/jobs/nonexistent-job-id-12345",
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 404
