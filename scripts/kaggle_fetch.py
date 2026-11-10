"""Download a Kaggle job's small result files (scores, metrics, configs, run info, log) into
outputs/kaggle/<name>/. Anomaly maps (.npz) are skipped by default: they stay on Kaggle.

Usage:
    python scripts/kaggle_fetch.py --name zsad-anomalyclip
    python scripts/kaggle_fetch.py --name zsad-anomalyclip --with-maps    # everything
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--with-maps", action="store_true")
    a = p.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    dest = REPO / "outputs" / "kaggle" / a.name
    dest.mkdir(parents=True, exist_ok=True)
    pattern = ".*" if a.with_maps else r"^(?!.*\.npz$).*"  # everything except .npz maps
    for attempt in range(5):  # the output listing is rate-limited (HTTP 429); back off and retry
        try:
            api.kernels_output(f"{user}/{a.name}", path=str(dest), file_pattern=pattern, quiet=True)
            break
        except Exception as e:
            if "429" not in str(e) or attempt == 4:
                raise
            time.sleep(30 * (attempt + 1))

    log = dest / f"{a.name}.log"
    if log.exists():  # Kaggle logs are JSON lines of {"stream_name", "data"}; keep a plain-text copy
        text = "".join(x.get("data", "") for x in json.loads(log.read_text(encoding="utf-8")))
        (dest / "log.txt").write_text(text, encoding="utf-8")
        print(text[-3000:].encode("ascii", "replace").decode())
    for m in sorted(dest.rglob("metrics.json")):
        r = json.loads(m.read_text(encoding="utf-8"))
        mean = {k: round(100 * v, 1) for k, v in r["mean"].items()}
        print(f"{m.parent.relative_to(dest)}: {mean}  complete={r['complete']}")


if __name__ == "__main__":
    main()
