"""Aggregate per-experiment results into CSVs.

`save_results_csv` merges new rows with any existing CSV so that successive
`--models X` invocations accumulate data rather than overwriting each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.common import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_results_csv(rows: list[dict[str, Any]], out_path: Path | str) -> Path:
    """Write new result rows to CSV, MERGING with any existing rows.

    If a CSV already exists at `out_path`, rows from the old CSV whose
    (model, question_id) pair also appears in `rows` are replaced; all
    other old rows are kept. This ensures that running
    `--models A` then `--models B` produces a CSV containing both A and B.
    """
    out = Path(out_path)
    ensure_dir(out.parent)
    new_df = pd.DataFrame(rows)

    if out.exists() and not new_df.empty:
        try:
            existing = pd.read_csv(out)
        except Exception:
            existing = pd.DataFrame()

        if (
            not existing.empty
            and "model" in existing.columns
            and "question_id" in existing.columns
            and "model" in new_df.columns
            and "question_id" in new_df.columns
        ):
            new_keys = set(zip(new_df["model"], new_df["question_id"]))
            mask = existing.apply(
                lambda r: (r["model"], str(r["question_id"])) not in new_keys
                and (r["model"], r["question_id"]) not in new_keys,
                axis=1,
            )
            kept = existing[mask]
            combined = pd.concat([kept, new_df], ignore_index=True)
        else:
            combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(out, index=False)
    logger.info("Wrote %d rows to %s (merged)", len(combined), out)
    return out


def summarize_by_model(csv_path: Path | str, numeric_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "model" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    agg = df.groupby("model")[cols].agg(["mean", "std"]).round(4)
    return agg
