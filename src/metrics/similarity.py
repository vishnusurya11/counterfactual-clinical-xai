"""Embedding similarity utilities (sentence-transformers wrapper)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def _get_model(name: str):
    from sentence_transformers import SentenceTransformer  # lazy import
    return SentenceTransformer(name)


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return embeddings @ embeddings.T


def mean_pairwise_similarity(embeddings: np.ndarray) -> float:
    n = embeddings.shape[0]
    if n < 2:
        return 1.0
    sim = cosine_sim_matrix(embeddings)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(sim[iu]))


def pair_similarity(text_a: str, text_b: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    emb = embed_texts([text_a, text_b], model_name)
    if emb.shape[0] < 2:
        return 0.0
    return float(emb[0] @ emb[1])
