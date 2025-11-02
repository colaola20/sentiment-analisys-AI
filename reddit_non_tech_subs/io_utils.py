from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from dotenv import load_dotenv as _dotenv_load


LOGGER = logging.getLogger(__name__)


def load_env() -> None:
    """Load environment from a .env if present, otherwise rely on os.environ."""
    _dotenv_load(override=False)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_now() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M")


def write_outputs(df: pd.DataFrame, outdir: str, base_name: str) -> Dict[str, str]:
    ensure_dir(outdir)
    ts = timestamp_now()
    csv_path = str(Path(outdir) / f"{base_name}_{ts}.csv")
    parquet_path = str(Path(outdir) / f"{base_name}_{ts}.parquet")
    df.to_csv(csv_path, index=True, quoting=csv.QUOTE_MINIMAL)
    # Prefer pyarrow if available
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        LOGGER.warning("Failed to write parquet with default engine: %s", e)
        try:
            df.to_parquet(parquet_path, index=False, engine="fastparquet")
        except Exception as e2:
            LOGGER.error("Failed to write parquet with fastparquet: %s", e2)
    return {"csv": csv_path, "parquet": parquet_path}


def append_run_log(outdir: str, meta: Dict[str, object]) -> None:
    ensure_dir(outdir)
    log_path = Path(outdir).parent / "RUN_LOG.md"
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    summary = {
        "timestamp_utc": ts,
        **meta,
    }
    line = f"- {ts} | subs={meta.get('subs_count')} | rows={meta.get('rows')} | mode={meta.get('mode')} | tf={meta.get('time_filter')} | hf={meta.get('use_hf')}\n"
    # Human-friendly markdown append
    if not log_path.exists():
        log_path.write_text("# Run Log\n\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
    # Also a JSONL for exact repro if needed
    jsonl_path = Path(outdir).parent / "RUN_LOG.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps(summary) + "\n")


