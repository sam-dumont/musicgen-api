"""Unit tests for the API endpoints."""

import os
import tempfile

# Set OUTPUT_DIR before importing app to avoid /data/output creation
os.environ["OUTPUT_DIR"] = tempfile.mkdtemp()

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gpu_available" in data


def test_metrics_endpoint(client):
    """Test metrics endpoint."""
    response = client.get("/metrics/json")
    assert response.status_code == 200
    data = response.json()
    assert "jobs_total" in data
    assert "jobs_completed" in data


def test_generate_requires_prompt(client):
    """Test that generate endpoint requires a prompt."""
    response = client.post("/generate", json={})
    assert response.status_code == 422  # Validation error


def test_separate_requires_input(client):
    """Test that separate endpoint requires audio input."""
    response = client.post("/separate")
    # Returns 400 when no file or URL provided
    assert response.status_code == 400
