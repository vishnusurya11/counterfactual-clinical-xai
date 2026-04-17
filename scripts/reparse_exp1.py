"""Re-parse all exp1_esi JSON files using the current (improved) parser.

This is a zero-cost, zero-API-call repair step. It walks every
`results/raw/exp1_esi/<model>/<qid>.json`, re-runs `parse_main_response()` on
each run's stored `response.text` (+ `response.thinking`), and overwrites the
`parsed` field.

Typical use after upgrading the parser:

    uv run python scripts/reparse_exp1.py               # dry-run report
    uv run python scripts/reparse_exp1.py --apply       # actually write
    uv run python scripts/reparse_exp1.py --apply --model deepseek-r1-distill-qwen-32b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import parse_main_response  # noqa: E402


def reparse_one(path: Path) -> tuple[int, int, int]:
    """Return (runs_total, runs_fixed, runs_still_bad)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    total = len(runs)
    fixed = 0
    still_bad = 0
    changed = False
    for r in runs:
        resp = r.get("response", {}) or {}
        text = resp.get("text", "") or ""
        thinking = resp.get("thinking", "") or ""
        new_parsed = parse_main_response(text, thinking=thinking)
        old_parsed = r.get("parsed", {}) or {}
        old_ok = bool(old_parsed.get("parse_ok"))
        new_ok = bool(new_parsed.get("parse_ok"))
        if old_parsed != new_parsed:
            r["parsed"] = new_parsed
            changed = True
        if not old_ok and new_ok:
            fixed += 1
        if not new_ok:
            still_bad += 1
    if changed:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return total, fixed, still_bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes to disk (default: dry run)")
    parser.add_argument("--model", default=None, help="limit to one model subdir")
    parser.add_argument("--exp", default="exp1_esi", help="experiment dir under results/raw (default: exp1_esi)")
    args = parser.parse_args()

    exp_root = ROOT / "results" / "raw" / args.exp
    if not exp_root.exists():
        print(f"ERROR: {exp_root} does not exist")
        return 1

    model_dirs = (
        [exp_root / args.model]
        if args.model
        else [d for d in sorted(exp_root.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    )

    grand_total = 0
    grand_fixed = 0
    grand_bad = 0
    grand_files = 0
    for mdir in model_dirs:
        if not mdir.exists():
            continue
        m_total = 0
        m_fixed = 0
        m_bad = 0
        m_files = 0
        for jf in sorted(mdir.glob("*.json")):
            m_files += 1
            if args.apply:
                t, f, b = reparse_one(jf)
            else:
                # Dry-run: count without writing
                data = json.loads(jf.read_text(encoding="utf-8"))
                runs = data.get("runs", [])
                t = len(runs)
                f = 0
                b = 0
                for r in runs:
                    resp = r.get("response", {}) or {}
                    new_parsed = parse_main_response(
                        resp.get("text", "") or "",
                        thinking=resp.get("thinking", "") or "",
                    )
                    old_ok = bool((r.get("parsed", {}) or {}).get("parse_ok"))
                    new_ok = bool(new_parsed.get("parse_ok"))
                    if not old_ok and new_ok:
                        f += 1
                    if not new_ok:
                        b += 1
            m_total += t
            m_fixed += f
            m_bad += b
        print(
            f"{mdir.name:42s}  files={m_files:3d}  runs={m_total:4d}  "
            f"fixed={m_fixed:4d}  still_bad={m_bad:4d}"
        )
        grand_total += m_total
        grand_fixed += m_fixed
        grand_bad += m_bad
        grand_files += m_files

    verb = "FIXED" if args.apply else "WOULD FIX"
    print(
        f"\nGRAND TOTAL: {grand_files} files, {grand_total} runs, "
        f"{verb} {grand_fixed}, still_bad {grand_bad}"
    )
    if not args.apply:
        print("\nDry run. Re-run with --apply to persist changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
