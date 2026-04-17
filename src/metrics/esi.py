"""Explanation Stability Index (ESI) computation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.metrics.similarity import embed_texts, mean_pairwise_similarity


def compute_esi(explanations: list[str], embedding_model: str = "all-MiniLM-L6-v2") -> float:
    """ESI = mean pairwise cosine similarity across N runs.

    Range: 0..1. 1.0 = perfectly consistent explanations.
    """
    if not explanations or len(explanations) < 2:
        return 1.0
    emb = embed_texts(explanations, embedding_model)
    return mean_pairwise_similarity(emb)


def answer_consistency(answers: list[str | None]) -> float:
    """Fraction of runs that return the modal answer."""
    clean = [a for a in answers if a]
    if not clean:
        return 0.0
    counts = Counter(clean)
    _, most_common_count = counts.most_common(1)[0]
    return most_common_count / len(clean)


def confidence_stats(confidences: list[int | None]) -> tuple[float, float]:
    clean = [c for c in confidences if isinstance(c, (int, float))]
    if not clean:
        return (0.0, 0.0)
    arr = np.asarray(clean, dtype=np.float32)
    return (float(arr.mean()), float(arr.std(ddof=0)))


def key_factor_jaccard(factor_lists: list[list[str]]) -> float:
    """Mean pairwise Jaccard similarity of KEY_FACTORS lists across runs."""
    sets = [set(f.lower().strip() for f in fl if f and isinstance(f, str)) for fl in factor_lists]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 1.0 if sets else 0.0
    sims: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            union = a | b
            if not union:
                sims.append(1.0)
            else:
                sims.append(len(a & b) / len(union))
    return float(np.mean(sims)) if sims else 0.0
