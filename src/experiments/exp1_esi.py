"""Experiment 1: Explanation Stability Index (ESI).

Run each question N times per model (temperature=0.7) and measure how
much the explanation varies across runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.experiments.common import call_and_wrap, ensure_dir, now_iso, save_json
from src.llm_client import LLMClient
from src.metrics.esi import answer_consistency, compute_esi, confidence_stats, key_factor_jaccard
from src.prompts import MAIN_PROMPT, format_options, parse_main_response
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_exp1(
    cfg: dict[str, Any],
    clients: dict[str, LLMClient],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exp_cfg = cfg["exp1_esi"]
    n_runs = int(exp_cfg["n_runs"])
    embedding_model = exp_cfg.get("embedding_model", "all-MiniLM-L6-v2")
    raw_root = ensure_dir(Path(cfg["storage"]["raw_dir"]) / "exp1_esi")

    all_results: list[dict[str, Any]] = []
    for model_name, client in clients.items():
        model_dir = ensure_dir(raw_root / model_name)
        logger.info("[exp1] model=%s n_questions=%d n_runs=%d", model_name, len(questions), n_runs)

        for q in tqdm(questions, desc=f"exp1:{model_name}", unit="q"):
            qid = q["id"]
            out_path = model_dir / f"{qid}.json"

            runs_data: list[dict[str, Any]] = []
            if out_path.exists():
                existing = _load_existing_runs(out_path)
                if len(existing) >= n_runs:
                    runs_data = existing[:n_runs]

            if len(runs_data) < n_runs:
                prompt = MAIN_PROMPT.format(question=q["question"], options=format_options(q["options"]))
                for run_idx in range(len(runs_data), n_runs):
                    resp_dict = call_and_wrap(client, prompt, run_idx=run_idx)
                    parsed = parse_main_response(
                        resp_dict["text"],
                        thinking=resp_dict.get("thinking", ""),
                    )
                    runs_data.append({
                        "run_idx": run_idx,
                        "response": resp_dict,
                        "parsed": parsed,
                    })

                save_json(out_path, {
                    "question_id": qid,
                    "model": model_name,
                    "question": q["question"],
                    "options": q["options"],
                    "correct_answer": q.get("correct_answer", ""),
                    "n_runs": n_runs,
                    "runs": runs_data,
                    "saved_at": now_iso(),
                })

            metrics = _aggregate_question(q, runs_data, embedding_model)
            metrics["model"] = model_name
            metrics["question_id"] = qid
            all_results.append(metrics)

    return all_results


def _load_existing_runs(path: Path) -> list[dict[str, Any]]:
    import json
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("runs", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _aggregate_question(
    q: dict[str, Any],
    runs: list[dict[str, Any]],
    embedding_model: str,
) -> dict[str, Any]:
    explanations = [r["parsed"].get("reasoning", "") for r in runs]
    answers = [r["parsed"].get("answer") for r in runs]
    confidences = [r["parsed"].get("confidence") for r in runs]
    factor_lists = [r["parsed"].get("key_factors", []) for r in runs]

    esi = compute_esi(explanations, embedding_model=embedding_model)
    ac = answer_consistency(answers)
    conf_mean, conf_std = confidence_stats(confidences)
    jacc = key_factor_jaccard(factor_lists)

    correct_letter = (q.get("correct_answer") or "").strip().upper()
    correct_rate = 0.0
    valid_answers = [a for a in answers if a]
    if valid_answers and correct_letter:
        correct_rate = sum(1 for a in valid_answers if a == correct_letter) / len(valid_answers)

    return {
        "esi_score": esi,
        "answer_consistency": ac,
        "confidence_mean": conf_mean,
        "confidence_std": conf_std,
        "key_factor_jaccard": jacc,
        "correct_rate": correct_rate,
    }
