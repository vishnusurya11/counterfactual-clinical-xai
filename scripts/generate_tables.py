"""Generate LaTeX tables from aggregated CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _load(agg_dir: Path, name: str) -> pd.DataFrame | None:
    path = agg_dir / name
    if not path.exists():
        logger.warning("Missing %s", path)
        return None
    return pd.read_csv(path)


def table1_main(agg_dir: Path, out_dir: Path, suffix: str) -> None:
    df1 = _load(agg_dir, f"exp1_esi_results{suffix}.csv")
    df2 = _load(agg_dir, f"exp2_ect_results{suffix}.csv")
    df3 = _load(agg_dir, f"exp3_pss_results{suffix}.csv")
    df4 = _load(agg_dir, f"exp4_counterfactual_results{suffix}.csv")
    df5 = _load(agg_dir, f"exp5_bias_results{suffix}.csv")

    rows = []
    for df, col, label in [
        (df1, "esi_score", "ESI"),
        (df1, "answer_consistency", "Answer Consistency"),
        (df1, "correct_rate", "Accuracy"),
        (df2, "cfs_score", "CFS"),
        (df3, "answer_stability", "PSS (answer)"),
        (df3, "explanation_stability", "PSS (explanation)"),
        (df4, "counterfactual_valid", "CF Validity"),
        (df4, "minimality", "CF Minimality"),
        (df4, "plausibility", "CF Plausibility"),
        (df5, "edd_score", "EDD (bias)"),
    ]:
        if df is None or col not in df.columns:
            continue
        agg = df.groupby("model")[col].agg(["mean", "std"])
        row = {"Metric": label}
        for model, s in agg.iterrows():
            row[model] = f"{s['mean']:.3f} ± {s['std']:.3f}"
        rows.append(row)

    if not rows:
        logger.warning("Table 1: no data")
        return

    table = pd.DataFrame(rows).set_index("Metric")
    latex = table.to_latex(column_format="l" + "c" * len(table.columns), escape=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"table1_main{suffix}.tex").write_text(latex, encoding="utf-8")
    (out_dir / f"table1_main{suffix}.csv").write_text(table.to_csv(), encoding="utf-8")
    logger.info("Wrote table1_main%s", suffix)


def table2_cited_unused(agg_dir: Path, out_dir: Path, suffix: str) -> None:
    df = _load(agg_dir, f"exp2_ect_results{suffix}.csv")
    if df is None or "cited_but_unused" not in df.columns:
        logger.warning("Table 2: no data")
        return

    from collections import Counter

    counter: Counter = Counter()
    for cell in df["cited_but_unused"].dropna():
        for item in str(cell).split(";"):
            item = item.strip().lower()
            if item:
                counter[item] += 1
    top = counter.most_common(20)
    table = pd.DataFrame(top, columns=["Concept", "Frequency"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"table2_cited_unused{suffix}.tex").write_text(table.to_latex(index=False), encoding="utf-8")
    (out_dir / f"table2_cited_unused{suffix}.csv").write_text(table.to_csv(index=False), encoding="utf-8")
    logger.info("Wrote table2_cited_unused%s", suffix)


def table3_counterfactual_examples(agg_dir: Path, raw_dir: Path, out_dir: Path, suffix: str) -> None:
    df = _load(agg_dir, f"exp4_counterfactual_results{suffix}.csv")
    if df is None:
        logger.warning("Table 3: no data")
        return
    sample = df.head(4)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"table3_counterfactual_examples{suffix}.csv").write_text(sample.to_csv(index=False), encoding="utf-8")
    (out_dir / f"table3_counterfactual_examples{suffix}.tex").write_text(sample.to_latex(index=False), encoding="utf-8")
    logger.info("Wrote table3_counterfactual_examples%s", suffix)


def main(poc: bool = False) -> None:
    cfg = load_config(ROOT / "config.yaml", poc=poc)
    agg_dir = ROOT / cfg["storage"]["aggregated_dir"]
    raw_dir = ROOT / cfg["storage"]["raw_dir"]
    out_dir = ROOT / "paper" / "tables"
    suffix = "_poc" if poc else ""

    table1_main(agg_dir, out_dir, suffix)
    table2_cited_unused(agg_dir, out_dir, suffix)
    table3_counterfactual_examples(agg_dir, raw_dir, out_dir, suffix)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--poc", action="store_true")
    args = p.parse_args()
    main(poc=args.poc)
