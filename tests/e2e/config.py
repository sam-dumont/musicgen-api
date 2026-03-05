"""E2E test configuration for MusicGen API."""

import os
import subprocess


def get_api_key_from_k8s() -> str | None:
    """Retrieve API key from Kubernetes secret."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "secret",
                "musicgen-api-key",
                "-n",
                "musicgen",
                "-o",
                "jsonpath={.data.api-key}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            import base64

            return base64.b64decode(result.stdout).decode("utf-8")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# API Configuration
API_URL = os.environ.get("MUSICGEN_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("MUSICGEN_API_KEY") or get_api_key_from_k8s()

# Timeouts and polling
JOB_POLL_INTERVAL = 5  # seconds between status checks
JOB_TIMEOUT = 300  # max wait time for job completion (5 minutes)
REQUEST_TIMEOUT = 30  # HTTP request timeout

# Test parameters
SHORT_DURATION = 10  # seconds for quick smoke tests
MEDIUM_DURATION = 30  # seconds for standard tests
