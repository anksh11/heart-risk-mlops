"""Send a stream of realistic (and a few invalid) requests to populate the dashboards."""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

import pandas as pd


def post(url: str, body: dict) -> int | str:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except OSError:  # connection reset/refused, e.g. while pods scale behind `minikube tunnel`
        return "conn_error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--rps", type=float, default=8, help="target requests/second per worker")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel workers (raise to trigger the HPA)")
    args = ap.parse_args()

    df = pd.read_csv("data/processed/heart_clean.csv").drop(columns=["target", "source"], errors="ignore")
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df.to_dict("records")]
    for r in rows:
        for k in ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"):
            if r[k] is not None:
                r[k] = int(r[k])

    end, codes, lock = time.time() + args.seconds, Counter(), threading.Lock()

    def worker() -> None:
        while time.time() < end:
            roll = random.random()
            if roll < 0.05:
                code = post(f"{args.url}/predict", {"age": 400})  # invalid -> 422
            elif roll < 0.15:
                code = post(f"{args.url}/predict/batch", {"patients": random.sample(rows, 5)})
            else:
                code = post(f"{args.url}/predict", random.choice(rows))
            with lock:
                codes[code] += 1
            time.sleep(1 / args.rps)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, args.concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"sent {sum(codes.values())} requests with {len(threads)} workers: {dict(codes)}")


if __name__ == "__main__":
    main()
