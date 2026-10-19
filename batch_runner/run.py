"""
Batch runner CLI tool.
Integration Standard #10: Runs a folder of test inputs against the API
and writes all results to one folder.

Usage:
    python batch_runner/run.py --input tests/inputs/ --output results/ --api http://localhost:8000
    python batch_runner/run.py --input tests/inputs/test1.json --output results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx


def run_batch(
    input_path: str,
    output_dir: str,
    api_base: str = "http://localhost:8000",
    timeout: int = 300,
    poll_interval: int = 5,
):
    """Run all test inputs against the API and save results."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = sorted(input_path.glob("*.json"))
    else:
        print(f"Error: {input_path} is not a file or directory")
        sys.exit(1)

    if not input_files:
        print(f"No .json files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(input_files)} test input(s)")
    print(f"API: {api_base}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    results = []
    client = httpx.Client(base_url=api_base, timeout=timeout)

    # Check API health
    try:
        health = client.get("/health")
        print(f"API health: {health.json()['status']}")
    except Exception as e:
        print(f"Error: API not reachable at {api_base}: {e}")
        sys.exit(1)

    for i, input_file in enumerate(input_files, 1):
        test_name = input_file.stem
        print(f"\n[{i}/{len(input_files)}] Running: {test_name}")

        with open(input_file) as f:
            request_body = json.load(f)

        test_output_dir = output_dir / test_name
        test_output_dir.mkdir(exist_ok=True)

        # Save request
        with open(test_output_dir / "request.json", "w") as f:
            json.dump(request_body, f, indent=2)

        try:
            # Submit job
            start_time = time.time()
            response = client.post("/api/v1/music/jobs", json=request_body)
            job_data = response.json()
            job_id = job_data["job_id"]
            print(f"  Job ID: {job_id}")

            # Poll for completion
            elapsed = 0
            while elapsed < timeout:
                status_response = client.get(f"/api/v1/music/jobs/{job_id}")
                status_data = status_response.json()
                status = status_data["status"]
                progress = status_data.get("progress_percent", 0)

                if status in ("completed", "failed"):
                    break

                print(f"  Status: {status} ({progress:.0f}%)", end="\r")
                time.sleep(poll_interval)
                elapsed = time.time() - start_time

            total_time = time.time() - start_time

            # Save response
            with open(test_output_dir / "response.json", "w") as f:
                json.dump(status_data, f, indent=2, default=str)

            result = {
                "test_name": test_name,
                "job_id": job_id,
                "status": status,
                "total_time_seconds": round(total_time, 2),
                "processing_time_seconds": status_data.get("processing_time_seconds"),
                "actual_duration_seconds": status_data.get("actual_duration_seconds"),
                "actual_loudness_lufs": status_data.get("actual_loudness_lufs"),
                "error": status_data.get("error"),
            }

            if status == "completed":
                print(f"  ✅ Completed in {total_time:.1f}s")
                # Download outputs if available
                if status_data.get("output_url"):
                    try:
                        audio_response = client.get(status_data["output_url"])
                        if audio_response.status_code == 200:
                            with open(test_output_dir / "mix.wav", "wb") as f:
                                f.write(audio_response.content)
                    except Exception:
                        pass
                if status_data.get("manifest_url"):
                    try:
                        manifest_response = client.get(status_data["manifest_url"])
                        if manifest_response.status_code == 200:
                            with open(test_output_dir / "music_manifest.json", "wb") as f:
                                f.write(manifest_response.content)
                    except Exception:
                        pass
            else:
                print(f"  ❌ {status}: {status_data.get('error', 'Unknown error')}")

            results.append(result)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "test_name": test_name,
                "status": "error",
                "error": str(e),
            })

    # Write summary
    summary = {
        "total_tests": len(results),
        "passed": sum(1 for r in results if r.get("status") == "completed"),
        "failed": sum(1 for r in results if r.get("status") != "completed"),
        "results": results,
    }

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results: {summary['passed']}/{summary['total_tests']} passed")
    print(f"Summary: {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch runner for Music Engine API")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file or folder")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--timeout", type=int, default=300, help="Max wait time per job (seconds)")
    parser.add_argument("--poll", type=int, default=5, help="Poll interval (seconds)")

    args = parser.parse_args()
    run_batch(args.input, args.output, args.api, args.timeout, args.poll)
