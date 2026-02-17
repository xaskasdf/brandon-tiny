"""Stress test for local LLM API - find max concurrent requests."""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_BASE = "http://localhost:5282"

PROMPT = "Explain what photosynthesis is in 3 sentences."
SYSTEM = "You are a helpful assistant. Be concise."


def single_request(worker_id: int):
    """Make a single API call, return (worker_id, latency, success, tokens)."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{API_BASE}/completion",
            json={
                "prompt": PROMPT,
                "system_prompt": SYSTEM,
                "temperature": 0.7,
                "max_tokens": 150,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        elapsed = time.perf_counter() - start
        return worker_id, elapsed, True, len(text.split())
    except Exception as e:
        elapsed = time.perf_counter() - start
        return worker_id, elapsed, False, 0


def run_batch(n_concurrent: int, requests_per_worker: int = 3):
    """Run n_concurrent workers, each making requests_per_worker requests."""
    total = n_concurrent * requests_per_worker
    print(f"\n{'='*60}")
    print(f"  {n_concurrent} concurrent workers x {requests_per_worker} requests = {total} total")
    print(f"{'='*60}")

    latencies = []
    successes = 0
    failures = 0
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = []
        for i in range(total):
            futures.append(pool.submit(single_request, i % n_concurrent))

        for fut in as_completed(futures):
            wid, lat, ok, tokens = fut.result()
            if ok:
                successes += 1
                latencies.append(lat)
            else:
                failures += 1

    wall_time = time.perf_counter() - start
    rps = successes / wall_time if wall_time > 0 else 0

    print(f"  Results:")
    print(f"    Success: {successes}/{total} ({100*successes/total:.0f}%)")
    print(f"    Failures: {failures}")
    print(f"    Wall time: {wall_time:.1f}s")
    print(f"    Throughput: {rps:.2f} req/s")
    if latencies:
        print(f"    Latency (avg): {statistics.mean(latencies):.2f}s")
        print(f"    Latency (p50): {statistics.median(latencies):.2f}s")
        print(f"    Latency (p95): {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
        print(f"    Latency (max): {max(latencies):.2f}s")

    return {
        "workers": n_concurrent,
        "successes": successes,
        "failures": failures,
        "wall_time": wall_time,
        "rps": rps,
        "avg_lat": statistics.mean(latencies) if latencies else 0,
    }


def main():
    # First verify the API is up
    print("Testing API connectivity...")
    try:
        _, lat, ok, _ = single_request(0)
        if ok:
            print(f"  API OK (latency: {lat:.2f}s)")
        else:
            print("  API returned error. Is the server running?")
            return
    except Exception as e:
        print(f"  API unreachable: {e}")
        return

    # Test increasing concurrency levels
    levels = [1, 2, 4, 8, 12, 16, 24, 32]
    results = []

    for n in levels:
        res = run_batch(n, requests_per_worker=3)
        results.append(res)

        # If failure rate > 30%, stop escalating
        total = res["successes"] + res["failures"]
        if res["failures"] / total > 0.3:
            print(f"\n  >> High failure rate at {n} workers, stopping escalation.")
            break

        # If throughput didn't improve much, note it
        if len(results) >= 2:
            prev_rps = results[-2]["rps"]
            curr_rps = res["rps"]
            if curr_rps < prev_rps * 0.8:
                print(f"\n  >> Throughput dropped ({prev_rps:.2f} -> {curr_rps:.2f} req/s)")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Workers':>8} | {'RPS':>8} | {'Avg Lat':>8} | {'Success':>8} | {'Wall':>8}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
    for r in results:
        total = r["successes"] + r["failures"]
        print(f"  {r['workers']:>8} | {r['rps']:>8.2f} | {r['avg_lat']:>7.2f}s | {r['successes']:>3}/{total:<3} | {r['wall_time']:>7.1f}s")

    # Find optimal
    best = max(results, key=lambda r: r["rps"])
    print(f"\n  >> Optimal: {best['workers']} workers @ {best['rps']:.2f} req/s")
    print(f"  >> Current generator uses 4 workers")
    if best["workers"] > 4:
        print(f"  >> Potential speedup: ~{best['rps']/results[0]['rps']:.1f}x over sequential")


if __name__ == "__main__":
    main()
