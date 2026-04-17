"""Counterfactual validity / minimality metrics."""

from __future__ import annotations

import difflib
from typing import Any


def edit_distance_ratio(a: str, b: str) -> float:
    """Ratio of changed tokens to total tokens. 0 = identical, 1 = totally different."""
    a_toks = (a or "").split()
    b_toks = (b or "").split()
    if not a_toks and not b_toks:
        return 0.0
    sm = difflib.SequenceMatcher(None, a_toks, b_toks)
    return 1.0 - sm.ratio()


def compute_counterfactual_metrics(
    original_answer: str | None,
    predicted_alternative: str | None,
    actual_answer_on_modified: str | None,
    original_question: str,
    modified_question: str,
    plausibility_score: int | None = None,
) -> dict[str, Any]:
    """Counterfactual validity/minimality/plausibility bundle.

    - counterfactual_validity: does the modified question actually produce
      the predicted alternative answer?
    - minimality_score: token-level edit distance ratio
      (lower = more minimal = "smallest change").
    - plausibility_score: supplied by an LLM-judge if available (1-5).
    """
    if predicted_alternative and actual_answer_on_modified:
        valid = (
            actual_answer_on_modified.strip().upper() == predicted_alternative.strip().upper()
            and actual_answer_on_modified.strip().upper() != (original_answer or "").strip().upper()
        )
    else:
        valid = False

    minimality = edit_distance_ratio(original_question, modified_question)

    return {
        "counterfactual_validity": bool(valid),
        "minimality_score": float(minimality),
        "plausibility_score": int(plausibility_score) if isinstance(plausibility_score, int) else None,
    }
