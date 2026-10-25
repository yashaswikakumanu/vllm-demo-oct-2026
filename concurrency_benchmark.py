
import csv
import json
import math
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

CONCURRENCY_LEVELS = [1,2,4,8]
REQUESTS_PER_LEVEL = 16
MAX_OUTPUT_TOKENS = 100

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Explain the main benefits of Kubernetes for "
    "deploying machine learning applications."
)

PAYLOAD = {
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": MAX_OUTPUT_TOKENS,
    "temperature": 0.0,
}

def percentile(values, percent):
    ordered = sorted(values)
    rank = math.ceil(len(ordered) * percent / 100)
    return ordered[max(0, rank - 1)]


def send_request(start_event):
    start_event.wait()

    request = urllib.request.Request(
        URL,
        data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request, timeout=180
        ) as response:
            result = json.loads(response.read())

        latency = time.perf_counter() - started

        return {
            "success": True,
            "latency_seconds": latency,
            "output_tokens": result["usage"]["completion_tokens"],
            "finish_reason": result["choices"][0]["finish_reason"],
            "error": "",
        }

    except Exception as exc:
        return {
            "success": False,
            "latency_seconds": time.perf_counter() - started,
            "output_tokens": 0,
            "finish_reason": "",
            "error": str(exc),
        }


def benchmark(concurrency):
    print(f"\nTesting concurrency = {concurrency}")

    results = []
    wall_start = time.perf_counter()

    # Each wave releases up to 'concurrency' requests together.
    for offset in range(0, REQUESTS_PER_LEVEL, concurrency):
        batch_size = min(
            concurrency, REQUESTS_PER_LEVEL - offset
        )
        start_event = Event()

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [
                pool.submit(send_request, start_event)
                for _ in range(batch_size)
            ]
            start_event.set()

            for future in futures:
                results.append(future.result())

    duration = time.perf_counter() - wall_start

    successful = [r for r in results if r["success"]]
    latencies = [r["latency_seconds"] for r in successful]
    output_tokens = sum(r["output_tokens"] for r in successful)

    summary = {
        "concurrency": concurrency,
        "requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "duration_seconds": round(duration, 3),
        "total_output_tokens": output_tokens,
        "requests_per_second": round(
            len(successful) / duration, 3
        ),
        "output_tokens_per_second": round(
            output_tokens / duration, 3
        ),
        "average_latency_seconds": (
            round(statistics.mean(latencies), 3)
            if latencies else None
        ),
        "p50_latency_seconds": (
            round(percentile(latencies, 50), 3)
            if latencies else None
        ),
        "p95_latency_seconds": (
            round(percentile(latencies, 95), 3)
            if latencies else None
        ),
    }

    print(json.dumps(summary, indent=2))

    for r in results:
        r["concurrency"] = concurrency

    return summary, results


def main():
    print("Warming up vLLM...")
    warmup_event = Event()
    warmup_event.set()
    warmup = send_request(warmup_event)

    if not warmup["success"]:
        raise RuntimeError(
            f"Warm-up failed: {warmup['error']}"
        )

    summaries = []
    details = []

    for concurrency in CONCURRENCY_LEVELS:
        summary, requests = benchmark(concurrency)
        summaries.append(summary)
        details.extend(requests)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    summary_path = RESULTS_DIR / f"summary_{timestamp}.csv"
    details_path = RESULTS_DIR / f"requests_{timestamp}.csv"
    metadata_path = RESULTS_DIR / f"metadata_{timestamp}.json"

    for path, rows in [
        (summary_path, summaries),
        (details_path, details),
    ]:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        "model": MODEL,
        "endpoint": URL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "requests_per_level": REQUESTS_PER_LEVEL,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "prompt": PROMPT,
        "benchmark_type": "closed batch waves",
        "note": (
            "Client-observed latency via kubectl port-forward. "
            "Output throughput includes all completed requests."
        ),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\nSaved results:")
    print(summary_path)
    print(details_path)
    print(metadata_path)


if __name__ == "__main__":
    main()

