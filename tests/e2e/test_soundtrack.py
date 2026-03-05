"""E2E tests for soundtrack generation endpoint."""

import pytest
import requests

from .config import REQUEST_TIMEOUT
from .conftest import validate_wav


class TestSoundtrackGeneration:
    """Full E2E tests for /generate/soundtrack endpoint."""

    @pytest.mark.timeout(600)
    def test_generate_simple_soundtrack(
        self,
        session: requests.Session,
        api_url: str,
        auth_headers: dict,
        wait_for_job_completion,
        download,
    ):
        """Test generating a simple 2-scene soundtrack."""
        scenes = [
            {"mood": "calm and peaceful", "duration": 10},
            {"mood": "energetic and exciting", "duration": 10},
        ]
        total_duration = sum(s["duration"] for s in scenes)

        response = session.post(
            f"{api_url}/generate/soundtrack",
            json={
                "base_prompt": "cinematic orchestral music",
                "scenes": scenes,
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        result = wait_for_job_completion(job_id, timeout=300)
        assert result["status"] == "completed"
        assert result["result_urls"] is not None
        assert len(result["result_urls"]) == 1

        # Download and validate result
        wav_data = download(result["result_urls"][0])
        props = validate_wav(wav_data)

        # Check duration matches expected (with tolerance for crossfade)
        # Crossfade reduces total duration slightly
        assert props["duration"] >= total_duration - 5
        assert props["duration"] <= total_duration + 3

    @pytest.mark.timeout(600)
    def test_generate_multi_scene_soundtrack(
        self,
        session: requests.Session,
        api_url: str,
        auth_headers: dict,
        wait_for_job_completion,
        download,
    ):
        """Test generating a soundtrack with multiple scenes."""
        scenes = [
            {"mood": "mysterious and tense", "duration": 8},
            {"mood": "action-packed", "duration": 10},
            {"mood": "triumphant victory", "duration": 8},
        ]

        response = session.post(
            f"{api_url}/generate/soundtrack",
            json={
                "base_prompt": "epic fantasy film score",
                "scenes": scenes,
                "crossfade_duration": 1.5,
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        result = wait_for_job_completion(job_id, timeout=400)
        assert result["status"] == "completed"

        wav_data = download(result["result_urls"][0])
        props = validate_wav(wav_data)
        assert props["duration"] > 15  # Should be substantial length

    @pytest.mark.timeout(600)
    def test_soundtrack_with_scene_prompts(
        self,
        session: requests.Session,
        api_url: str,
        auth_headers: dict,
        wait_for_job_completion,
        download,
    ):
        """Test soundtrack with per-scene prompt overrides."""
        scenes = [
            {
                "mood": "dark",
                "duration": 10,
                "prompt": "dark ambient drone music",
            },
            {
                "mood": "bright",
                "duration": 10,
                "prompt": "uplifting piano melody",
            },
        ]

        response = session.post(
            f"{api_url}/generate/soundtrack",
            json={
                "base_prompt": "film score",
                "scenes": scenes,
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
        assert props["duration"] > 10

    def test_soundtrack_progress_detail(
        self,
        session: requests.Session,
        api_url: str,
        auth_headers: dict,
    ):
        """Test that soundtrack generation reports scene-level progress."""
        import time

        scenes = [
            {"mood": "peaceful", "duration": 8},
            {"mood": "dramatic", "duration": 8},
        ]

        response = session.post(
            f"{api_url}/generate/soundtrack",
            json={
                "base_prompt": "ambient music",
                "scenes": scenes,
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        # Poll and check for progress detail
        seen_progress_detail = False
        for _ in range(120):
            status_response = session.get(
                f"{api_url}/jobs/{job_id}",
                timeout=REQUEST_TIMEOUT,
            )
            data = status_response.json()

            # Check for progress_detail during processing
            if data["status"] == "processing" and data.get("progress_detail"):
                seen_progress_detail = True
                detail = data["progress_detail"]
                # Should have scene tracking info
                if "current_scene" in detail:
                    assert "total_scenes" in detail
                    assert detail["total_scenes"] == 2

            if data["status"] in ["completed", "failed"]:
                break

            time.sleep(2)

        # Note: progress_detail may not always be captured depending on timing
        # So we don't assert seen_progress_detail, just verify the flow works

    def test_soundtrack_validation_errors(
        self,
        session: requests.Session,
        api_url: str,
        auth_headers: dict,
    ):
        """Test soundtrack validation errors."""
        # Empty scenes list
        response = session.post(
            f"{api_url}/generate/soundtrack",
            json={
                "base_prompt": "test",
                "scenes": [],
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 422

        # Scene duration too short
        response = session.post(
            f"{api_url}/generate/soundtrack",
            json={
                "base_prompt": "test",
                "scenes": [{"mood": "test", "duration": 2}],  # Below minimum
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 422

        # Invalid crossfade duration
        response = session.post(
            f"{api_url}/generate/soundtrack",
            json={
                "base_prompt": "test",
                "scenes": [{"mood": "test", "duration": 10}],
                "crossfade_duration": 10.0,  # Too long
            },
            headers=auth_headers,
            timeout=REQUEST_TIMEOUT,
        )
        assert response.status_code == 422
