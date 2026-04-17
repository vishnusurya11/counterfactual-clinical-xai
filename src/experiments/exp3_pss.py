"""Experiment 3: Perturbation Stability Score (PSS).

Paraphrase each question N times, run original + paraphrases through each
model, and measure whether answers/explanations stay stable under
semantically equivalent perturbations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.experiments.common import call_and_wrap, ensure_dir, now_iso, save_json
from src.llm_client import LLMClient
from src.metrics.pss import compute_pss
from src.metrics.similarity import pair_similarity
from src.prompts import MAIN_PROMPT, PERTURBATION_PROMPT, format_options, parse_main_response
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_exp3(
    cfg: dict[str, Any],
    clients: dict[str, LLMClient],
    questions: list[dict[str, Any]],
    *,
    helper_clients: dict[str, LLMClient] | None = None,
) -> list[dict[str, Any]]:
    exp_cfg = cfg["exp3_pss"]
    n_perturbations = int(exp_cfg["n_perturbations"])
    perturb_model = exp_cfg.get("perturbation_model", "gpt-4o-mini")
    min_sim = float(exp_cfg.get("min_semantic_similarity", 0.85))

    raw_root = ensure_dir(Path(cfg["storage"]["raw_dir"]) / "exp3_pss")
    perturbation_cache = ensure_dir(raw_root / "_paraphrases")

    # Paraphrase generator must always route to its configured model
    helper_pool = helper_clients if helper_clients is not None else clients
    paraphrase_client = helper_pool.get(perturb_model)
    if paraphrase_client is None:
        logger.warning(
            "perturbation_model '%s' not available in helper_clients %s; falling back to first available",
            perturb_model, list(helper_pool.keys()),
        )
        paraphrase_client = next(iter(helper_pool.values()))
    logger.info("[exp3] paraphrase generator routed to: %s", paraphrase_client.name)

    # Stage A: generate paraphrases once per question (reused across all models)
    paraphrases_by_qid: dict[str, list[str]] = {}
    for q in tqdm(questions, desc="exp3:paraphrase", unit="q"):
        qid = q["id"]
        pfile = perturbation_cache / f"{qid}.json"
        if pfile.exists():
            import json
            cached = json.loads(pfile.read_text(encoding="utf-8"))
            paraphrases_by_qid[qid] = cached.get("paraphrases", [])
            continue

        collected: list[str] = []
        for i in range(n_perturbations):
            prompt = PERTURBATION_PROMPT.format(question=q["question"])
            resp = call_and_wrap(paraphrase_client, prompt, run_idx=i, temperature=0.3)
            candidate = resp["text"].strip()
            sim = pair_similarity(q["question"], candidate)
            if sim >= min_sim:
                collected.append(candidate)
            else:
                logger.warning("[exp3] paraphrase similarity %.2f < %.2f for %s; keeping anyway", sim, min_sim, qid)
                collected.append(candidate)

        paraphrases_by_qid[qid] = collected
        save_json(pfile, {"question_id": qid, "original": q["question"], "paraphrases": collected})

    # Stage B: for each model, answer original + paraphrases
    all_results: list[dict[str, Any]] = []
    for model_name, client in clients.items():
        model_dir = ensure_dir(raw_root / model_name)
        logger.info("[exp3] model=%s n_questions=%d n_perturbations=%d", model_name, len(questions), n_perturbations)

        for q in tqdm(questions, desc=f"exp3:{model_name}", unit="q"):
            qid = q["id"]
            out_path = model_dir / f"{qid}.json"

            # Skip if this question already has a result — resume-safe
            if out_path.exists():
                try:
                    existing = json.loads(out_path.read_text(encoding="utf-8"))
                    if existing.get("metrics"):
                        all_results.append({
                            "model": model_name,
                            "question_id": qid,
                            **existing["metrics"],
                        })
                        continue
                except Exception:
                    pass

            try:
                original_prompt = MAIN_PROMPT.format(question=q["question"], options=format_options(q["options"]))
                original_resp = call_and_wrap(client, original_prompt, run_idx=0)
                original_parsed = parse_main_response(
                    original_resp["text"],
                    thinking=original_resp.get("thinking", ""),
                )

                paraphrase_results = []
                for pi, ptext in enumerate(paraphrases_by_qid.get(qid, [])):
                    p_prompt = MAIN_PROMPT.format(question=ptext, options=format_options(q["options"]))
                    p_resp = call_and_wrap(client, p_prompt, run_idx=pi + 1)
                    p_parsed = parse_main_response(
                        p_resp["text"],
                        thinking=p_resp.get("thinking", ""),
                    )
                    paraphrase_results.append({
                        "paraphrase_idx": pi,
                        "paraphrased_question": ptext,
                        "response": p_resp,
                        "parsed": p_parsed,
                    })

                metrics = compute_pss(
                    original={
                        "answer": original_parsed.get("answer"),
                        "reasoning": original_parsed.get("reasoning", ""),
                        "key_factors": original_parsed.get("key_factors", []),
                        "confidence": original_parsed.get("confidence"),
                    },
                    paraphrased=[
                        {
                            "answer": r["parsed"].get("answer"),
                            "reasoning": r["parsed"].get("reasoning", ""),
                            "key_factors": r["parsed"].get("key_factors", []),
                            "confidence": r["parsed"].get("confidence"),
                        }
                        for r in paraphrase_results
                    ],
                )

                save_json(out_path, {
                    "question_id": qid,
                    "model": model_name,
                    "original_question": q["question"],
                    "original_response": original_resp,
                    "original_parsed": original_parsed,
                    "paraphrase_results": paraphrase_results,
                    "metrics": metrics,
                    "saved_at": now_iso(),
                })

                all_results.append({
                    "model": model_name,
                    "question_id": qid,
                    **metrics,
                })
            except Exception as e:
                logger.warning("[exp3] %s/%s failed: %s — skipping, will retry on re-run", model_name, qid, e)

    return all_results
