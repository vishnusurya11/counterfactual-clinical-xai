"""Experiment 4: Counterfactual Validity.

Ask each model to generate a counterfactual: the smallest clinical change
that would flip the correct answer. Then test whether the modified question
actually produces the predicted alternative answer when re-asked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.experiments.common import call_and_wrap, ensure_dir, now_iso, save_json
from src.llm_client import LLMClient
from src.metrics.counterfactual import compute_counterfactual_metrics
from src.prompts import (
    COUNTERFACTUAL_PROMPT,
    MAIN_PROMPT,
    PLAUSIBILITY_JUDGE_PROMPT,
    format_options,
    parse_counterfactual_response,
    parse_main_response,
    parse_plausibility_response,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_exp4(
    cfg: dict[str, Any],
    clients: dict[str, LLMClient],
    questions: list[dict[str, Any]],
    *,
    helper_clients: dict[str, LLMClient] | None = None,
) -> list[dict[str, Any]]:
    exp_cfg = cfg["exp4_counterfactual"]
    n_q = int(exp_cfg["n_questions"])
    judge_model = exp_cfg.get("plausibility_judge", "gpt-4o-mini")
    raw_root = ensure_dir(Path(cfg["storage"]["raw_dir"]) / "exp4_counterfactual")
    exp1_root = Path(cfg["storage"]["raw_dir"]) / "exp1_esi"

    sub = questions[:n_q]

    # Judge must always route to its configured model (e.g. gpt-4o-mini via OpenRouter),
    # NEVER fall back to the model under test. A small-context model like MedGemma-27B
    # at 4096 ctx can't handle the judge prompt (5400+ tokens).
    helper_pool = helper_clients if helper_clients is not None else clients
    judge = helper_pool.get(judge_model)
    if judge is None:
        logger.warning(
            "plausibility_judge '%s' not available in helper_clients %s; falling back to first available",
            judge_model, list(helper_pool.keys()),
        )
        judge = next(iter(helper_pool.values()))
    logger.info("[exp4] plausibility judge routed to: %s", judge.name)

    all_results: list[dict[str, Any]] = []
    for model_name, client in clients.items():
        model_dir = ensure_dir(raw_root / model_name)
        logger.info("[exp4] model=%s n_questions=%d", model_name, len(sub))

        for q in tqdm(sub, desc=f"exp4:{model_name}", unit="q"):
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
                            "counterfactual_valid": existing["metrics"].get("counterfactual_validity"),
                            "minimality": existing["metrics"].get("minimality_score"),
                            "plausibility": existing["metrics"].get("plausibility_score"),
                            "change_description": (existing.get("counterfactual_parsed") or {}).get("change", ""),
                        })
                        continue
                except Exception:
                    pass

            try:
                exp1_path = exp1_root / model_name / f"{qid}.json"
                orig_reasoning, orig_answer = _load_first_run(exp1_path)
                if not orig_reasoning or not orig_answer:
                    logger.warning("[exp4] no exp1 data for %s/%s — skipping", model_name, qid)
                    continue

                option_keys = sorted(q["options"].keys())
                alternative = None
                for k in option_keys:
                    if k != orig_answer:
                        alternative = k
                        break
                if alternative is None:
                    continue

                cf_prompt = COUNTERFACTUAL_PROMPT.format(
                    answer=orig_answer,
                    question=q["question"],
                    reasoning=orig_reasoning,
                    alternative_answer=alternative,
                )
                cf_resp = call_and_wrap(client, cf_prompt)
                cf_parsed = parse_counterfactual_response(cf_resp["text"])

                modified_q = cf_parsed.get("modified_question", "")
                if not modified_q.strip():
                    modified_q = q["question"]

                verify_prompt = MAIN_PROMPT.format(question=modified_q, options=format_options(q["options"]))
                verify_resp = call_and_wrap(client, verify_prompt)
                verify_parsed = parse_main_response(
                    verify_resp["text"],
                    thinking=verify_resp.get("thinking", ""),
                )

                judge_prompt = PLAUSIBILITY_JUDGE_PROMPT.format(
                    original_question=q["question"],
                    modified_question=modified_q,
                    change_description=cf_parsed.get("change", ""),
                )
                judge_resp = call_and_wrap(judge, judge_prompt)
                judge_parsed = parse_plausibility_response(judge_resp["text"])

                metrics = compute_counterfactual_metrics(
                    original_answer=orig_answer,
                    predicted_alternative=alternative,
                    actual_answer_on_modified=verify_parsed.get("answer"),
                    original_question=q["question"],
                    modified_question=modified_q,
                    plausibility_score=judge_parsed.get("score"),
                )

                save_json(out_path, {
                    "question_id": qid,
                    "model": model_name,
                    "original_answer": orig_answer,
                    "predicted_alternative": alternative,
                    "counterfactual_response": cf_resp,
                    "counterfactual_parsed": cf_parsed,
                    "verify_response": verify_resp,
                    "verify_parsed": verify_parsed,
                    "judge_response": judge_resp,
                    "judge_parsed": judge_parsed,
                    "metrics": metrics,
                    "saved_at": now_iso(),
                })

                all_results.append({
                    "model": model_name,
                    "question_id": qid,
                    "counterfactual_valid": metrics["counterfactual_validity"],
                    "minimality": metrics["minimality_score"],
                    "plausibility": metrics["plausibility_score"],
                    "change_description": cf_parsed.get("change", ""),
                })
            except Exception as e:
                logger.warning("[exp4] %s/%s failed: %s — skipping, will retry on re-run", model_name, qid, e)

    return all_results


def _load_first_run(exp1_path: Path) -> tuple[str, str | None]:
    if not exp1_path.exists():
        return "", None
    import json
    try:
        data = json.loads(exp1_path.read_text(encoding="utf-8"))
    except Exception:
        return "", None
    runs = data.get("runs", [])
    if not runs:
        return "", None
    # Iterate runs to find the first one with a usable (reasoning, answer) pair.
    # Re-parses from response.text + response.thinking to pick up parser upgrades.
    for r in runs:
        if not isinstance(r, dict):
            continue
        resp = r.get("response", {}) or {}
        text = resp.get("text", "") or ""
        thinking = resp.get("thinking", "") or ""
        new_parsed = parse_main_response(text, thinking=thinking)
        if new_parsed.get("reasoning") and new_parsed.get("answer"):
            return new_parsed["reasoning"], new_parsed["answer"]
        stored = r.get("parsed", {}) or {}
        if stored.get("reasoning") and stored.get("answer"):
            return stored["reasoning"], stored["answer"]
    return "", None
