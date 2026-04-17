"""Standalone progress-grid for the 5 experiments × 6 models.

Walks results/raw/exp*/<model>/*.json and prints a compact grid showing
how many per-question JSON files each cell contains, with targets pulled
from config.yaml. Read-only, safe to run concurrently with a live
main.py run.

Usage:
    uv run python scripts/status_grid.py              # one-shot
    uv run python scripts/status_grid.py --watch 30   # auto-refresh
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402


EXPERIMENTS = [
    ("exp1_esi",              "exp1_esi",           "data"),
    ("exp2_ect",              "exp2_ect",           "data"),
    ("exp3_pss",              "exp3_pss",           "data"),
    ("exp4_counterfactual",   "exp4_counterfactual", "exp4_counterfactual"),
    ("exp5_bias",             "exp5_bias",          "exp5_bias"),
]


def _target_for(exp_key: str, cfg: dict) -> int:
    """Per-experiment target question count pulled from config.yaml."""
    if exp_key in ("exp1_esi", "exp2_ect", "exp3_pss"):
        return int(cfg["data"]["n_questions"])
    if exp_key == "exp4_counterfactual":
        return int(cfg.get("exp4_counterfactual", {}).get("n_questions", cfg["data"]["n_questions"]))
    if exp_key == "exp5_bias":
        return int(cfg.get("exp5_bias", {}).get("n_questions", cfg["data"]["n_questions"]))
    return int(cfg["data"]["n_questions"])


def _count_files(raw_root: Path, exp_dir: str, model: str) -> int:
    d = raw_root / exp_dir / model
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.is_file() and p.suffix == ".json")


def _render(cfg: dict) -> str:
    raw_root = ROOT / cfg["storage"]["raw_dir"]
    models = [m["name"] for m in cfg["models"]]

    # Determine column widths
    exp_col = max(len(e[0]) for e in EXPERIMENTS) + 2
    model_cols = {m: max(len(m), 5) for m in models}
    target_col = 6

    # Header
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"  counterfactual-clinical-xai  |  {cfg['storage']['raw_dir']}  |  {ts}"
    width = exp_col + sum(model_cols.values()) + len(models) * 3 + target_col + 3
    bar = "=" * max(width, len(title) + 2)

    lines: list[str] = []
    lines.append(bar)
    lines.append(title)
    lines.append(bar)

    header_cells = [f"{'Experiment':<{exp_col}}"]
    for m in models:
        header_cells.append(f"{m:>{model_cols[m]}}")
    header_cells.append(f"{'target':>{target_col}}")
    lines.append(" | ".join(header_cells))

    sep_cells = ["-" * exp_col]
    for m in models:
        sep_cells.append("-" * model_cols[m])
    sep_cells.append("-" * target_col)
    lines.append("-|-".join(sep_cells))

    # Data rows
    for exp_name, exp_dir, _src in EXPERIMENTS:
        target = _target_for(exp_dir, cfg)
        row_cells = [f"{exp_name:<{exp_col}}"]
        for m in models:
            count = _count_files(raw_root, exp_dir, m)
            width_m = model_cols[m]
            if count >= target and target > 0:
                cell = f"{count}*"
            elif count == 0:
                cell = f"{count}"
            else:
                cell = f"{count}"
            row_cells.append(f"{cell:>{width_m}}")
        row_cells.append(f"{target:>{target_col}}")
        lines.append(" | ".join(row_cells))

    lines.append(bar)

    # Legend + totals
    total_done = 0
    total_target = 0
    for exp_name, exp_dir, _ in EXPERIMENTS:
        target = _target_for(exp_dir, cfg)
        for m in models:
            total_target += target
            total_done += min(_count_files(raw_root, exp_dir, m), target)
    pct = (total_done / total_target * 100) if total_target else 0.0
    lines.append(f"  Overall: {total_done} / {total_target} cells filled  ({pct:.1f}%)   * = at-or-above target")
    lines.append(bar)

    return "\n".join(lines)


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def main() -> int:
    parser = argparse.ArgumentParser(description="Progress grid for the 5 experiments × 6 models")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                        help="refresh every N seconds (Ctrl+C to exit)")
    parser.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    try:
        cfg = load_config(cfg_path, poc=False)
    except Exception as e:
        print(f"Failed to load config at {cfg_path}: {e}")
        return 1

    if args.watch > 0:
        try:
            while True:
                _clear_screen()
                print(_render(cfg))
                print(f"\n(refreshing every {args.watch}s — Ctrl+C to exit)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nexit")
            return 0

    print(_render(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
