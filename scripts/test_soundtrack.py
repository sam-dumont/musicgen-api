#!/usr/bin/env python3
"""Test script for multi-mood soundtrack generation.

Usage:
    # Start the server first:
    make run-reload

    # Then in another terminal:
    python test_soundtrack.py

    # Or with custom server URL:
    python test_soundtrack.py --url http://localhost:8000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def test_soundtrack(base_url: str = "http://localhost:8000", api_key: str = None):
    """Generate a 1-minute test soundtrack with 3 mood transitions."""

    print("=" * 60)
    print("Multi-Mood Soundtrack Generation Test")
    print("=" * 60)
    print()

    # Test request: 1 minute with 3 scenes (20s each)
    payload = {
        "base_prompt": "cinematic orchestral score with strings and piano",
        "scenes": [
            {
                "mood": "mysterious and tense, building suspense",
                "duration": 20,
            },
            {
                "mood": "energetic and triumphant, heroic brass",
                "duration": 20,
            },
            {
                "mood": "peaceful and nostalgic, gentle resolution",
                "duration": 20,
            },
        ],
        "use_beat_aligned_crossfade": True,
        "crossfade_duration": 2.0,
    }

    total_duration = sum(s["duration"] for s in payload["scenes"])
    print(f"Configuration:")
    print(f"  Total duration: {total_duration} seconds")
    print(f"  Scenes: {len(payload['scenes'])}")
    print(f"  Beat-aligned crossfade: {payload['use_beat_aligned_crossfade']}")
    print(f"  Crossfade duration: {payload['crossfade_duration']}s")
    print()

    for i, scene in enumerate(payload["scenes"], 1):
        print(f"  Scene {i}: {scene['mood'][:40]}... ({scene['duration']}s)")
    print()

    # Headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    # Submit job
    print("Submitting soundtrack generation job...")
    try:
        response = requests.post(
            f"{base_url}/generate/soundtrack",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to server at {base_url}")
        print("Make sure the server is running: make run-reload")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nError: {e}")
        print(f"Response: {response.text}")
        sys.exit(1)

    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"Job submitted: {job_id}")
    print()

    # Poll for completion
    print("Generating soundtrack (this may take several minutes)...")
    print("-" * 60)

    start_time = time.time()
    last_scene = 0
    last_stage = ""

    while True:
        response = requests.get(f"{base_url}/jobs/{job_id}", timeout=30)
        status_data = response.json()

        status = status_data["status"]
        progress = status_data.get("progress", 0) or 0
        progress_detail = status_data.get("progress_detail") or {}

        current_scene = progress_detail.get("current_scene", 0)
        total_scenes = progress_detail.get("total_scenes", 0)
        stage = progress_detail.get("stage", "")

        # Print updates on changes
        if current_scene != last_scene or stage != last_stage:
            elapsed = time.time() - start_time
            if stage == "generating" and current_scene:
                scene_mood = payload["scenes"][current_scene - 1]["mood"][:30]
                print(f"  [{elapsed:5.1f}s] Scene {current_scene}/{total_scenes}: {scene_mood}...")
            elif stage == "crossfading":
                print(f"  [{elapsed:5.1f}s] Crossfading segments...")
            elif stage == "finalizing":
                print(f"  [{elapsed:5.1f}s] Finalizing...")
            last_scene = current_scene
            last_stage = stage

        if status == "completed":
            elapsed = time.time() - start_time
            print("-" * 60)
            print(f"Completed in {elapsed:.1f} seconds!")
            print()

            # Download result
            result_urls = status_data.get("result_urls", [])
            if result_urls:
                for url in result_urls:
                    filename = url.split("/")[-1]
                    print(f"Downloading: {filename}")

                    file_response = requests.get(f"{base_url}{url}", timeout=60)
                    output_path = Path("output") / filename
                    output_path.parent.mkdir(exist_ok=True)

                    with open(output_path, "wb") as f:
                        f.write(file_response.content)

                    size_mb = len(file_response.content) / (1024 * 1024)
                    print(f"Saved to: {output_path} ({size_mb:.2f} MB)")
            break

        elif status == "failed":
            print("-" * 60)
            print(f"Job failed: {status_data.get('error', 'Unknown error')}")
            sys.exit(1)

        time.sleep(2)

    print()
    print("=" * 60)
    print("Test completed successfully!")
    print()
    print("Listen to the output file and verify:")
    print("  1. Smooth transitions between moods at ~20s and ~40s")
    print("  2. No audible clicks or pops at transition points")
    print("  3. Musical coherence maintained throughout")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test multi-mood soundtrack generation")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key if required",
    )
    args = parser.parse_args()

    test_soundtrack(args.url, args.api_key)


if __name__ == "__main__":
    main()
