"""Wait until Kaggle jobs finish, printing their status. A just-pushed kernel can return HTTP 404
for a few minutes; that is treated as "pending", not as finished.

A kernel that stays QUEUED for more than --max-queue-min minutes is reported as "STUCK_QUEUED"
(we've seen the API report QUEUED for hours with nothing in the queue; re-pushing fixed it).

Usage:
    python scripts/kaggle_wait.py zsad-repro-anomalyclip zsad-repro-anomalyvfm
    python scripts/kaggle_wait.py zsad-bench-clipfam --max-queue-min 30
"""

from __future__ import annotations

import argparse
import time

FINAL = ("COMPLETE", "ERROR", "CANCEL")  # KernelWorkerStatus values that mean "finished"


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--max-queue-min", type=float, default=None)
    a = p.parse_args()
    names = a.names
    start = time.time()
    finished: dict[str, str] = {}
    while len(finished) < len(names):
        for n in names:
            if n in finished:
                continue
            try:
                status = str(api.kernels_status(f"{user}/{n}").status).split(".")[-1]
            except Exception as e:  # 404 right after pushing, or a transient network error
                status = f"PENDING ({str(e)[:40]})"
            if status in FINAL:
                finished[n] = status
            elif status == "QUEUED" and a.max_queue_min and time.time() - start > 60 * a.max_queue_min:
                finished[n] = status = "STUCK_QUEUED"
            print(time.strftime("%H:%M"), n, status, flush=True)
        if len(finished) < len(names):
            time.sleep(120)
    print("ALL FINISHED:", finished)


if __name__ == "__main__":
    main()
