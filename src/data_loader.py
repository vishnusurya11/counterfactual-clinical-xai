"""MedQA data loader.

Downloads MedQA (USMLE subset) from HuggingFace, selects N questions,
and saves as JSONL. Also provides demographic-variant generation for Exp5.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


DEMOGRAPHIC_TEMPLATES = {
    "no_demographic": None,
    "male_white": "A {age}-year-old white male",
    "female_white": "A {age}-year-old white female",
    "male_black": "A {age}-year-old Black male",
    "female_black": "A {age}-year-old Black female",
    "male_hispanic": "A {age}-year-old Hispanic male",
    "female_hispanic": "A {age}-year-old Hispanic female",
}

# Patterns that existing MedQA questions use to open with demographics
_DEMO_OPENER = re.compile(
    r"^\s*A\s+(?P<age>\d+)[\s-]?year[\s-]?old\s+(?P<desc>[^.]*?)\s+(?:man|woman|male|female|boy|girl)[^.]*?\.",
    re.IGNORECASE,
)


def load_medqa(
    hf_dataset_id: str,
    hf_subset: str,
    split: str,
    n_questions: int,
    seed: int,
    output_path: Path | str,
    pool_path: Path | str,
) -> list[dict[str, Any]]:
    """Load N MedQA questions using a frozen ID pool + append-only selected set.

    - `pool_path` holds a once-built, deterministic shuffled index ordering of
      the full dataset (keyed by seed/dataset/subset/split).
    - `output_path` is an append-only JSONL of selected records, each tagged
      with `selection_order` and `_pool_offset` for full provenance.

    Asking for more records than currently selected extends the JSONL by
    pulling the next indices from the pool — existing records are never
    touched or re-ordered.
    """
    output_path = Path(output_path)
    pool_path = Path(pool_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.parent.mkdir(parents=True, exist_ok=True)

    pool = _load_or_build_pool(
        pool_path=pool_path,
        hf_dataset_id=hf_dataset_id,
        hf_subset=hf_subset,
        split=split,
        seed=seed,
    )
    ordered_indices: list[int] = pool["ordered_indices"]

    existing = _read_selected(output_path)
    if len(existing) >= n_questions:
        logger.info(
            "Using %d cached selected questions from %s (requested %d)",
            len(existing), output_path, n_questions,
        )
        return existing[:n_questions]

    next_pool_offset = (max((r["_pool_offset"] for r in existing), default=-1) + 1)
    next_order = len(existing)
    need = n_questions - len(existing)
    logger.info(
        "Extending selection: have %d, need %d more (pool cursor=%d)",
        len(existing), need, next_pool_offset,
    )

    from datasets import load_dataset  # lazy import (heavy)

    logger.info("Loading %s (%s, split=%s) for extension", hf_dataset_id, hf_subset, split)
    try:
        ds = load_dataset(hf_dataset_id, hf_subset, split=split, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("Primary dataset load failed (%s); trying fallback 'openlifescienceai/medqa'", e)
        ds = load_dataset("openlifescienceai/medqa", split=split)

    appended = 0
    with output_path.open("a", encoding="utf-8") as f:
        cursor = next_pool_offset
        while cursor < len(ordered_indices) and appended < need:
            ds_idx = ordered_indices[cursor]
            row = ds[ds_idx]
            record = _normalize_row(row, ds_idx)
            if record is None:
                logger.debug("skipping unnormalizable row at pool_offset=%d ds_idx=%d", cursor, ds_idx)
                cursor += 1
                continue
            record["selection_order"] = next_order
            record["_pool_offset"] = cursor
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing.append(record)
            next_order += 1
            appended += 1
            cursor += 1

    if appended < need:
        logger.warning(
            "Pool exhausted: wanted %d more but only appended %d (pool size=%d)",
            need, appended, len(ordered_indices),
        )

    logger.info("Selected %d MedQA questions total (added %d)", len(existing), appended)
    return existing[:n_questions]


def _load_or_build_pool(
    pool_path: Path,
    hf_dataset_id: str,
    hf_subset: str,
    split: str,
    seed: int,
) -> dict[str, Any]:
    """Return the frozen pool metadata, building it on first call."""
    if pool_path.exists():
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        mismatches = []
        for key, expected in (
            ("dataset", hf_dataset_id),
            ("subset", hf_subset),
            ("split", split),
            ("seed", seed),
        ):
            if pool.get(key) != expected:
                mismatches.append(f"{key}: pool={pool.get(key)!r} config={expected!r}")
        if mismatches:
            raise RuntimeError(
                "medqa_pool.json does not match current config:\n  "
                + "\n  ".join(mismatches)
                + f"\nTo regenerate: delete {pool_path} AND the selected JSONL, then rerun."
            )
        logger.info("Loaded frozen pool from %s (%d indices)", pool_path, len(pool["ordered_indices"]))
        return pool

    from datasets import load_dataset  # lazy import (heavy)

    logger.info("Building frozen pool: downloading %s (%s, split=%s)", hf_dataset_id, hf_subset, split)
    try:
        ds = load_dataset(hf_dataset_id, hf_subset, split=split, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("Primary dataset load failed (%s); trying fallback 'openlifescienceai/medqa'", e)
        ds = load_dataset("openlifescienceai/medqa", split=split)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    pool = {
        "dataset": hf_dataset_id,
        "subset": hf_subset,
        "split": split,
        "seed": seed,
        "dataset_size": len(ds),
        "ordered_indices": indices,
    }
    pool_path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote frozen pool: %d indices → %s", len(indices), pool_path)
    return pool


def _read_selected(path: Path) -> list[dict[str, Any]]:
    """Read the append-only selected JSONL, sorted by selection_order."""
    if not path.exists():
        return []
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for r in records:
        if "selection_order" not in r or "_pool_offset" not in r:
            raise RuntimeError(
                f"{path} has record(s) missing 'selection_order' / '_pool_offset' — "
                "this file predates the frozen-pool scheme. Delete it (and the pool file) to regenerate."
            )
    records.sort(key=lambda r: r["selection_order"])
    return records


def _normalize_row(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Normalize various MedQA schemas to a common shape."""
    qid = row.get("id") or row.get("question_id") or f"medqa_{idx}"
    question = row.get("question") or row.get("sent1") or ""
    if not question:
        return None

    options_raw = row.get("options") or row.get("choices") or row.get("option")
    correct = row.get("answer_idx") or row.get("answer") or row.get("correct") or ""

    options: dict[str, str] = {}
    if isinstance(options_raw, list):
        # list of {"key": "A", "value": "..."} OR list of strings
        for i, opt in enumerate(options_raw):
            key = chr(ord("A") + i)
            if isinstance(opt, dict):
                k = opt.get("key") or opt.get("label") or key
                v = opt.get("value") or opt.get("text") or opt.get("option") or ""
                options[k] = v
            else:
                options[key] = str(opt)
    elif isinstance(options_raw, dict):
        for k, v in options_raw.items():
            options[k.upper()] = v if isinstance(v, str) else str(v)

    if not options:
        return None

    # Normalize correct answer to a single letter
    correct_letter = ""
    if isinstance(correct, str):
        correct_letter = correct.strip()[:1].upper() if correct.strip() else ""
    if not correct_letter and "answer_idx" in row:
        try:
            correct_letter = chr(ord("A") + int(row["answer_idx"]))
        except Exception:  # noqa: BLE001
            correct_letter = ""

    return {
        "id": str(qid),
        "question": question.strip(),
        "options": options,
        "correct_answer": correct_letter,
        "meta": {k: row[k] for k in row if k not in {"question", "options", "choices", "answer", "answer_idx", "id"}},
    }


def apply_demographic_variant(question: str, variant: str) -> str:
    """Rewrite the opening line of a question to use a demographic variant."""
    if variant == "no_demographic":
        # Strip existing demographic opener if any
        return _DEMO_OPENER.sub(
            lambda m: f"A patient presents",
            question,
            count=1,
        ).strip()

    template = DEMOGRAPHIC_TEMPLATES.get(variant)
    if template is None:
        return question

    # Try to preserve the detected age from the original question
    m = _DEMO_OPENER.search(question)
    age = m.group("age") if m else "45"
    prefix = template.format(age=age)

    if m:
        rest = question[m.end():].lstrip()
        return f"{prefix} presents {rest}"
    return f"{prefix} presents with {question[0].lower()}{question[1:]}"
