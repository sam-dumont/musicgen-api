"""Smoke tests for MusicGen API - quick validation that the API is responsive."""

import pytest
import requests

from .config import REQUEST_TIMEOUT, SHORT_DURATION


class TestHealth:
    """Health and metrics endpoint tests."""

    def test_health_endpoint(self, session: requests.Session, api_url: str):
        """Test that health endpoint returns healthy status."""
        response = session.get(f"{api_url}/health", timeout=REQUEST_TIMEOUT)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "gpu_available" in data
        assert "mps_available" in data
        assert "device" in data
        assert "models_loaded" in data

    def test_metrics_endpoint(self, session: requests.Session, api_url: str):
        """Test that metrics endpoint returns valid structure."""
        response = session.get(f"{api_url}/metrics/json", timeout=REQUEST_TIMEOUT)
        assert response.status_code == 200

        data = response.json()
        assert "jobs_total" in data
        assert "jobs_completed" in data
        assert "jobs_failed" in data
        assert "jobs_in_progress" in data


class TestAuthentication:
    """API key authentication tests."""

    def test_generate_requires_auth(self, session: requests.Session, api_url: str):
        """Test that generate endpoint requires API key."""
        response = session.post(
            f"{api_url}/generate",
            json={"prompt": "test", "duration": 10},
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 401
        assert "Missing X-API-Key" in response.json()["detail"]

    def test_generate_rejects_invalid_key(self, session: requests.Session, api_url: str):
        """Test that generate endpoint rejects invalid API key."""
        response = session.post(
            f"{api_url}/generate",
            json={"prompt": "test", "duration": 10},
            headers={"X-API-Key": "invalid-key-12345"},
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 403
        assert "Invalid API key" in response.json()["detail"]

    def test_separate_requires_auth(self, session: requests.Session, api_url: str):
        """Test that separate endpoint requires API key."""
        response = session.post(
            f"{api_url}/separate",
            json={"audio_url": "https://example.com/test.mp3"},
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 401

    def test_soundtrack_requires_auth(self, session: requests.Session, api_url: str):
        """Test that soundtrack endpoint requires API key."""
        response = session.post(
            f"{api_url}/generate/soundtrack",
            json={
                "base_prompt": "test",
                "scenes": [{"mood": "happy", "duration": 10}],
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 401


class TestValidation:
    """Input validation tests."""

    def test_generate_requires_prompt(
        self, session: requests.Session, api_url: str, auth_headers: dict
    ):
        """Test that generate endpoint requires a prompt."""
        response = session.post(
            f"{api_url}/generate",
            json={"duration": 10},
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 422

    def test_generate_validates_duration_min(
        self, session: requests.Session, api_url: str, auth_headers: dict
    ):
        """Test that generate rejects duration below minimum."""
        response = session.post(
            f"{api_url}/generate",
            json={"prompt": "test music", "duration": 5},
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 422

    def test_generate_validates_duration_max(
        self, session: requests.Session, api_url: str, auth_headers: dict
    ):
        """Test that generate rejects duration above maximum."""
        response = session.post(
            f"{api_url}/generate",
            json={"prompt": "test music", "duration": 200},
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 422

    def test_separate_requires_input(
        self, session: requests.Session, api_url: str, auth_headers: dict
    ):
        """Test that separate endpoint requires audio input."""
        response = session.post(
            f"{api_url}/separate",
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 400

    def test_soundtrack_requires_scenes(
        self, session: requests.Session, api_url: str, auth_headers: dict
    ):
        """Test that soundtrack endpoint requires scenes."""
        response = session.post(
            f"{api_url}/generate/soundtrack",
            json={"base_prompt": "test"},
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 422


class TestQuickGeneration:
    """Quick generation test to verify the pipeline works."""

    def test_submit_generation_job(
        self, session: requests.Session, api_url: str, auth_headers: dict
    ):
        """Test that a generation job can be submitted successfully."""
        response = session.post(
            f"{api_url}/generate",
            json={
                "prompt": "calm ambient electronic music",
                "duration": SHORT_DURATION,
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200

        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        # Job ID should be a valid UUID format
        assert len(data["job_id"]) == 36

    def test_job_status_endpoint(
        self, session: requests.Session, api_url: str, auth_headers: dict
    ):
        """Test that job status can be queried."""
        # Submit a job first
        response = session.post(
            f"{api_url}/generate",
            json={
                "prompt": "quick test music",
                "duration": SHORT_DURATION,
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Query status with short timeout (just verify endpoint responds)
        import time

        time.sleep(1)  # Brief wait

        status_response = session.get(
            f"{api_url}/jobs/{job_id}",
            timeout=60,  # Longer timeout for status check
        )
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ["queued", "processing", "completed", "failed"]
        assert "progress" in status_data
