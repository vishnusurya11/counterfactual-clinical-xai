"""Experiment 5: Demographic Bias in Explanations.

For questions where demographics should NOT affect diagnosis, insert
demographic variants and measure whether explanations change.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data_loader import apply_demographic_variant
from src.experiments.common import call_and_wrap, ensure_dir, now_iso, save_json
from src.llm_client import LLMClient
from src.metrics.similarity import embed_texts, mean_pairwise_similarity, pair_similarity
from src.prompts import (
    DEMOGRAPHIC_VARIANT_PROMPT,
    STEREOTYPE_JUDGE_PROMPT,
    format_options,
    parse_main_response,
    parse_stereotype_response,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_exp5(
    cfg: dict[str, Any],
    clients: dict[str, LLMClient],
    questions: list[dict[str, Any]],
    *,
    helper_clients: dict[str, LLMClient] | None = None,
) -> list[dict[str, Any]]:
    exp_cfg = cfg["exp5_bias"]
    n_q = int(exp_cfg["n_questions"])
    variants = list(exp_cfg["demographic_variants"])
    judge_model = exp_cfg.get("stereotype_judge", "gpt-4o-mini")
    raw_root = ensure_dir(Path(cfg["storage"]["raw_dir"]) / "exp5_bias")

    sub = questions[:n_q]

    # Judge must always route to its configured model regardless of --models filter.
    helper_pool = helper_clients if helper_clients is not None else clients
    judge = helper_pool.get(judge_model)
    if judge is None:
        logger.warning(
            "stereotype_judge '%s' not available in helper_clients %s; falling back to first available",
            judge_model, list(helper_pool.keys()),
        )
        judge = next(iter(helper_pool.values()))
    logger.info("[exp5] stereotype judge routed to: %s", judge.name)

    all_results: list[dict[str, Any]] = []
    for model_name, client in clients.items():
        model_dir = ensure_dir(raw_root / model_name)
        logger.info("[exp5] model=%s n_questions=%d n_variants=%d", model_name, len(sub), len(variants))

        for q in tqdm(sub, desc=f"exp5:{model_name}", unit="q"):
            qid = q["id"]
            out_path = model_dir / f"{qid}.json"

            # Skip if this question already has a result — resume-safe
            if out_path.exists():
                try:
                    existing = json.loads(out_path.read_text(encoding="utf-8"))
                    if existing.get("metrics"):
                        m = existing["metrics"]
                        baseline_answer = (
                            (existing.get("variants") or {})
                            .get("no_demographic", {})
                            .get("parsed", {})
                            .get("answer")
                        )
                        all_results.append({
                            "model": model_name,
                            "question_id": qid,
                            "edd_score": m.get("edd_score"),
                            "answer_changed": m.get("answer_changed"),
                            "n_stereotype_flags": m.get("n_stereotype_flags", 0),
                            "baseline_answer": baseline_answer,
                        })
                        continue
                except Exception:
                    pass

            try:
                variant_results: dict[str, dict[str, Any]] = {}
                for vi, variant in enumerate(variants):
                    demographic_question = apply_demographic_variant(q["question"], variant)
                    prompt = DEMOGRAPHIC_VARIANT_PROMPT.format(
                        question_with_demographic=demographic_question,
                        options=format_options(q["options"]),
                    )
                    resp = call_and_wrap(client, prompt, run_idx=vi)
                    parsed = parse_main_response(resp["text"], thinking=resp.get("thinking", ""))
                    variant_results[variant] = {
                        "demographic_question": demographic_question,
                        "response": resp,
                        "parsed": parsed,
                    }

                explanations = [v["parsed"].get("reasoning", "") for v in variant_results.values()]
                emb = embed_texts(explanations)
                mean_sim = mean_pairwise_similarity(emb)
                edd_score = 1.0 - mean_sim

                answers = {k: v["parsed"].get("answer") for k, v in variant_results.items()}
                baseline_answer = answers.get("no_demographic")
                any_changed = any(a != baseline_answer for a in answers.values() if a)

                stereotype_flags: dict[str, dict[str, Any]] = {}
                baseline_exp = variant_results.get("no_demographic", {}).get("parsed", {}).get("reasoning", "")
                for variant, vr in variant_results.items():
                    if variant == "no_demographic":
                        continue
                    judge_prompt = STEREOTYPE_JUDGE_PROMPT.format(
                        exp_a=baseline_exp,
                        demographic=variant,
                        exp_b=vr["parsed"].get("reasoning", ""),
                    )
                    judge_resp = call_and_wrap(judge, judge_prompt)
                    judge_parsed = parse_stereotype_response(judge_resp["text"])
                    stereotype_flags[variant] = {
                        "judge_response": judge_resp,
                        "parsed": judge_parsed,
                    }

                n_stereotypes = sum(1 for s in stereotype_flags.values() if s["parsed"].get("stereotype_found"))

                save_json(out_path, {
                    "question_id": qid,
                    "model": model_name,
                    "variants": variant_results,
                    "metrics": {
                        "edd_score": edd_score,
                        "answer_changed": any_changed,
                        "n_stereotype_flags": n_stereotypes,
                    },
                    "stereotype_flags": stereotype_flags,
                    "saved_at": now_iso(),
                })

                all_results.append({
                    "model": model_name,
                    "question_id": qid,
                    "edd_score": edd_score,
                    "answer_changed": any_changed,
                    "n_stereotype_flags": n_stereotypes,
                    "baseline_answer": baseline_answer,
                })
            except Exception as e:
                logger.warning("[exp5] %s/%s failed: %s — skipping, will retry on re-run", model_name, qid, e)

    return all_results
