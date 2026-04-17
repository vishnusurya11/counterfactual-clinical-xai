"""Perturbation Stability Score (PSS)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.metrics.esi import key_factor_jaccard
from src.metrics.similarity import embed_texts, mean_pairwise_similarity


def compute_pss(original: dict[str, Any], paraphrased: list[dict[str, Any]]) -> dict[str, float]:
    """Compute stability under paraphrase perturbations.

    original / paraphrased entries have keys: answer, reasoning,
    key_factors (list), confidence (int|None).
    """
    all_runs = [original] + list(paraphrased)

    # Answer stability
    answers = [r.get("answer") for r in all_runs]
    ref = original.get("answer")
    if ref is None or not paraphrased:
        answer_stability = 0.0
    else:
        same = sum(1 for a in [p.get("answer") for p in paraphrased] if a == ref)
        answer_stability = same / len(paraphrased)

    # Explanation stability — mean pairwise cosine of all explanations
    reasonings = [r.get("reasoning", "") or "" for r in all_runs]
    emb = embed_texts(reasonings)
    explanation_stability = mean_pairwise_similarity(emb)

    # Concept (key_factor) stability
    factor_lists = [r.get("key_factors", []) or [] for r in all_runs]
    concept_stability = key_factor_jaccard(factor_lists)

    # Confidence drift
    confs = [r.get("confidence") for r in all_runs if isinstance(r.get("confidence"), (int, float))]
    if len(confs) >= 2:
        ref_c = confs[0]
        drifts = [abs(c - ref_c) for c in confs[1:]]
        confidence_drift = float(np.mean(drifts)) if drifts else 0.0
    else:
        confidence_drift = 0.0

    return {
        "answer_stability": float(answer_stability),
        "explanation_stability": float(explanation_stability),
        "concept_stability": float(concept_stability),
        "confidence_drift": float(confidence_drift),
    }
