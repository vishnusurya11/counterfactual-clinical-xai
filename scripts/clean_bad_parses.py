"""Remove runs with parse_ok=False from per-question JSON files.

Use after fixing a max_tokens / prompt issue so the runner re-generates
only the broken runs on the next invocation. Cache auto-invalidates on
the new max_tokens so fresh calls actually happen.

Usage:
    uv run python scripts/clean_bad_parses.py                    # dry run
    uv run python scripts/clean_bad_parses.py --apply            # actually delete
    uv run python scripts/clean_bad_parses.py --apply --model medgemma-27b
    uv run python scripts/clean_bad_parses.py --apply --exp exp1_esi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def scan_file(path: Path) -> tuple[int, int]:
    """Return (n_bad, n_total) without modifying the file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0
    runs = data.get("runs", [])
    bad = sum(1 for r in runs if not r.get("parsed", {}).get("parse_ok", True))
    return bad, len(runs)


def clean_file(path: Path) -> tuple[int, int]:
    """Strip bad runs from the file. Returns (n_removed, n_remaining)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    good = [r for r in runs if r.get("parsed", {}).get("parse_ok", True)]
    removed = len(runs) - len(good)
    if removed > 0:
        data["runs"] = good
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return removed, len(good)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="actually modify files")
    p.add_argument("--exp", default=None, help="limit to one experiment dir (e.g. exp1_esi)")
    p.add_argument("--model", default=None, help="limit to one model")
    args = p.parse_args()

    exp_dirs = [RAW / args.exp] if args.exp else [d for d in RAW.iterdir() if d.is_dir() and d.name.startswith("exp")]

    total_bad = 0
    total_files_touched = 0
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        model_dirs = [exp_dir / args.model] if args.model else [d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
        for model_dir in model_dirs:
            if not model_dir.exists():
                continue
            for jf in sorted(model_dir.glob("*.json")):
                bad, total = scan_file(jf)
                if bad == 0:
                    continue
                total_bad += bad
                total_files_touched += 1
                if args.apply:
                    removed, remaining = clean_file(jf)
                    print(f"{jf.relative_to(ROOT)}: removed {removed}/{total}, now has {remaining} good runs")
                else:
                    print(f"[dry] {jf.relative_to(ROOT)}: would remove {bad}/{total} bad runs")

    verb = "removed" if args.apply else "would remove"
    print(f"\n{verb} {total_bad} bad runs across {total_files_touched} files")
    if not args.apply:
        print("Re-run with --apply to actually modify files.")


if __name__ == "__main__":
    sys.exit(main())
