#!/usr/bin/env python3
"""Generate test soundtracks simulating what immich-memory-generator sends.

This creates 2 test tracks of ~60 seconds each with realistic family video
memory prompts and mood transitions.

Usage:
    # Start the server first:
    make run-reload

    # Then run this script:
    python generate_soundtracks.py
"""

import argparse
import sys
import time
from pathlib import Path

import requests


def generate_simple(
    base_url: str,
    name: str,
    prompt: str,
    duration: int,
    api_key: str | None = None,
) -> str | None:
    """Generate using the simple /generate endpoint (no scenes/moods)."""

    print(f"\n{'='*60}")
    print(f"Generating (SIMPLE): {name}")
    print(f"{'='*60}")

    print(f"Duration: {duration}s")
    print(f"Prompt: {prompt[:60]}...")
    print()

    payload = {
        "prompt": prompt,
        "duration": duration,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    # Submit job
    print("Submitting job...")
    try:
        response = requests.post(
            f"{base_url}/generate",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to server at {base_url}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"\nError: {e}")
        print(f"Response: {response.text}")
        return None

    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"Job submitted: {job_id}")

    # Poll for completion
    print("\nGenerating...")
    print("-" * 40)

    start_time = time.time()
    while True:
        response = requests.get(f"{base_url}/jobs/{job_id}", timeout=120)
        status_data = response.json()
        status = status_data["status"]
        progress = status_data.get("progress", 0)

        elapsed = time.time() - start_time
        print(f"\r  [{elapsed:5.1f}s] Progress: {progress:.0f}%", end="", flush=True)

        if status == "completed":
            print()
            print("-" * 40)
            print(f"Completed in {elapsed:.1f}s")

            result_urls = status_data.get("result_urls", [])
            if result_urls:
                for url in result_urls:
                    file_response = requests.get(f"{base_url}{url}", timeout=60)

                    output_dir = Path("test_outputs")
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"{name}.wav"

                    with open(output_path, "wb") as f:
                        f.write(file_response.content)

                    size_mb = len(file_response.content) / (1024 * 1024)
                    print(f"Saved: {output_path} ({size_mb:.2f} MB)")
                    return str(output_path)
            break

        elif status == "failed":
            print()
            print("-" * 40)
            print(f"Job failed: {status_data.get('error', 'Unknown error')}")
            return None

        time.sleep(2)

    return None


def generate_soundtrack(
    base_url: str,
    name: str,
    base_prompt: str,
    scenes: list[dict],
    api_key: str | None = None,
) -> str | None:
    """Generate using the /generate/soundtrack endpoint (with scenes)."""

    print(f"\n{'='*60}")
    print(f"Generating (SOUNDTRACK): {name}")
    print(f"{'='*60}")

    total_duration = sum(s["duration"] for s in scenes)
    print(f"Total duration: {total_duration}s | Scenes: {len(scenes)}")
    print(f"Base prompt: {base_prompt[:60]}...")
    print()

    for i, scene in enumerate(scenes, 1):
        mood = scene.get("mood", "(no mood)")
        print(f"  Scene {i}: {mood[:35]}... ({scene['duration']}s)")

    payload = {
        "base_prompt": base_prompt,
        "scenes": scenes,
        "use_beat_aligned_crossfade": True,
        "crossfade_duration": 2.0,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    # Submit job
    print(f"\nSubmitting job...")
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
        print("Start the server with: make run-reload")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"\nError: {e}")
        print(f"Response: {response.text}")
        return None

    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"Job submitted: {job_id}")

    # Poll for completion
    print("\nGenerating (this may take several minutes)...")
    print("-" * 40)

    start_time = time.time()
    last_scene = 0
    last_stage = ""

    while True:
        response = requests.get(f"{base_url}/jobs/{job_id}", timeout=120)
        status_data = response.json()

        status = status_data["status"]
        progress_detail = status_data.get("progress_detail") or {}

        current_scene = progress_detail.get("current_scene", 0)
        total_scenes = progress_detail.get("total_scenes", 0)
        stage = progress_detail.get("stage", "")

        if current_scene != last_scene or stage != last_stage:
            elapsed = time.time() - start_time
            if stage == "generating" and current_scene:
                scene_mood = (scenes[current_scene - 1].get("mood") or "(no mood)")[:25]
                print(f"  [{elapsed:5.1f}s] Scene {current_scene}/{total_scenes}: {scene_mood}...")
            elif stage == "crossfading":
                print(f"  [{elapsed:5.1f}s] Crossfading segments...")
            elif stage == "finalizing":
                print(f"  [{elapsed:5.1f}s] Finalizing...")
            last_scene = current_scene
            last_stage = stage

        if status == "completed":
            elapsed = time.time() - start_time
            print("-" * 40)
            print(f"Completed in {elapsed:.1f}s")

            result_urls = status_data.get("result_urls", [])
            if result_urls:
                for url in result_urls:
                    filename = url.split("/")[-1]
                    file_response = requests.get(f"{base_url}{url}", timeout=60)

                    # Save to test_outputs directory
                    output_dir = Path("test_outputs")
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"{name}.wav"

                    with open(output_path, "wb") as f:
                        f.write(file_response.content)

                    size_mb = len(file_response.content) / (1024 * 1024)
                    print(f"Saved: {output_path} ({size_mb:.2f} MB)")
                    return str(output_path)
            break

        elif status == "failed":
            print("-" * 40)
            print(f"Job failed: {status_data.get('error', 'Unknown error')}")
            return None

        time.sleep(2)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate test soundtracks simulating immich-memory-generator"
    )
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

    print("=" * 60)
    print("COMPARISON TEST: /generate vs /generate/soundtrack")
    print("=" * 60)
    print()
    print("Testing SAME prompt and duration with both endpoints to")
    print("isolate if the soundtrack generator is causing silence.")
    print()

    prompt = (
        "acoustic guitar and gentle piano, "
        "warm cinematic family video soundtrack, "
        "no vocals, clean production"
    )
    duration = 70

    results = []

    # Test 1: Simple /generate endpoint (max 120s, no scenes/moods)
    print(">>> TEST 1: Using /generate endpoint (simple, no scenes)")
    result = generate_simple(
        base_url=args.url,
        name="test_70s_simple",
        prompt=prompt,
        duration=duration,
        api_key=args.api_key,
    )
    if result:
        results.append(("simple /generate", result))
    else:
        print("\nFailed to generate with /generate endpoint")
        sys.exit(1)

    # Test 2: Soundtrack endpoint with single scene, NO mood
    print("\n>>> TEST 2: Using /generate/soundtrack endpoint (no mood)")
    scenes = [{"duration": duration}]  # No mood field
    result = generate_soundtrack(
        base_url=args.url,
        name="test_70s_soundtrack_no_mood",
        base_prompt=prompt,
        scenes=scenes,
        api_key=args.api_key,
    )
    if result:
        results.append(("soundtrack (no mood)", result))
    else:
        print("\nFailed to generate with /generate/soundtrack endpoint")
        sys.exit(1)

    print()
    print("=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)
    print()
    print("Generated files to compare:")
    for label, path in results:
        print(f"  - {label}: {path}")
    print()
    print("Listen to both and compare:")
    print("  - Does /generate have audio throughout?")
    print("  - Does /generate/soundtrack have silence?")
    print("  - If soundtrack has silence, the issue is in soundtrack.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
