"""Experiment 2: Explanation Causality Test (ECT).

Extract the model's stated key concepts from its explanation, then ablate
each concept and see if the answer changes. If the answer flips, the
concept is causally important. If not, it was "cited but unused" — evidence
of post-hoc rationalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.experiments.common import call_and_wrap, ensure_dir, now_iso, save_json
from src.llm_client import LLMClient
from src.metrics.ect import compute_ect
from src.prompts import (
    CONCEPT_ABLATION_PROMPT,
    CONCEPT_EXTRACTION_PROMPT,
    format_options,
    parse_concept_list,
    parse_main_response,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_exp2(
    cfg: dict[str, Any],
    clients: dict[str, LLMClient],
    questions: list[dict[str, Any]],
    *,
    helper_clients: dict[str, LLMClient] | None = None,
) -> list[dict[str, Any]]:
    exp_cfg = cfg["exp2_ect"]
    n_concepts = int(exp_cfg["n_concepts_to_ablate"])
    concept_extract_model = exp_cfg.get("concept_extraction_model", "gpt-4o-mini")
    raw_root = ensure_dir(Path(cfg["storage"]["raw_dir"]) / "exp2_ect")
    exp1_root = Path(cfg["storage"]["raw_dir"]) / "exp1_esi"

    # Helper (concept extractor) must always route to its configured model
    # (e.g. gpt-4o-mini via OpenRouter), never fall back to a filtered client.
    helper_pool = helper_clients if helper_clients is not None else clients
    extractor = helper_pool.get(concept_extract_model)
    if extractor is None:
        logger.warning(
            "concept_extraction_model '%s' not available in helper_clients %s; falling back to first available",
            concept_extract_model, list(helper_pool.keys()),
        )
        extractor = next(iter(helper_pool.values()))
    logger.info("[exp2] concept extractor routed to: %s", extractor.name)

    all_results: list[dict[str, Any]] = []
    for model_name, client in clients.items():
        model_dir = ensure_dir(raw_root / model_name)
        logger.info("[exp2] model=%s n_questions=%d", model_name, len(questions))

        for q in tqdm(questions, desc=f"exp2:{model_name}", unit="q"):
            qid = q["id"]
            out_path = model_dir / f"{qid}.json"

            # Skip if this question already has a result — resume-safe
            if out_path.exists():
                try:
                    existing = json.loads(out_path.read_text(encoding="utf-8"))
                    if existing.get("metrics"):
                        row = {
                            "model": model_name,
                            "question_id": qid,
                            "n_concepts": existing["metrics"].get("n_concepts", 0),
                            "n_causal": existing["metrics"].get("n_causal", 0),
                            "cfs_score": existing["metrics"].get("causal_faithfulness_score", 0.0),
                            "cited_but_unused": "; ".join(existing["metrics"].get("cited_but_unused", [])),
                            "genuinely_causal": "; ".join(existing["metrics"].get("genuinely_causal", [])),
                        }
                        all_results.append(row)
                        continue
                except Exception:
                    pass  # fall through to re-run

            try:
                exp1_path = exp1_root / model_name / f"{qid}.json"
                original_reasoning, original_answer = _load_first_run(exp1_path)
                if not original_reasoning or not original_answer:
                    logger.warning("[exp2] no exp1 data for %s/%s — skipping", model_name, qid)
                    continue

                extract_prompt = CONCEPT_EXTRACTION_PROMPT.format(reasoning=original_reasoning)
                extract_resp = call_and_wrap(extractor, extract_prompt)
                concepts = parse_concept_list(extract_resp["text"])[:n_concepts]

                ablations: dict[str, dict[str, Any]] = {}
                for concept in concepts:
                    ablate_prompt = CONCEPT_ABLATION_PROMPT.format(
                        concept=concept,
                        question=q["question"],
                        options=format_options(q["options"]),
                    )
                    ablate_resp = call_and_wrap(client, ablate_prompt)
                    parsed = parse_main_response(
                        ablate_resp["text"],
                        thinking=ablate_resp.get("thinking", ""),
                    )
                    ablations[concept] = {
                        "response": ablate_resp,
                        "parsed": parsed,
                    }

                ablated_answers = {c: a["parsed"].get("answer") for c, a in ablations.items()}
                metrics = compute_ect(original_answer, ablated_answers)

                save_json(out_path, {
                    "question_id": qid,
                    "model": model_name,
                    "original_answer": original_answer,
                    "original_reasoning": original_reasoning,
                    "concepts": concepts,
                    "extraction_response": extract_resp,
                    "ablations": ablations,
                    "metrics": metrics,
                    "saved_at": now_iso(),
                })

                row = {
                    "model": model_name,
                    "question_id": qid,
                    "n_concepts": metrics["n_concepts"],
                    "n_causal": metrics["n_causal"],
                    "cfs_score": metrics["causal_faithfulness_score"],
                    "cited_but_unused": "; ".join(metrics["cited_but_unused"]),
                    "genuinely_causal": "; ".join(metrics["genuinely_causal"]),
                }
                all_results.append(row)
            except Exception as e:
                logger.warning("[exp2] %s/%s failed: %s — skipping, will retry on re-run", model_name, qid, e)

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
    # Re-parses from response.text + response.thinking to pick up any parser
    # upgrades since the runs were first written.
    for r in runs:
        if not isinstance(r, dict):
            continue
        resp = r.get("response", {}) or {}
        text = resp.get("text", "") or ""
        thinking = resp.get("thinking", "") or ""
        new_parsed = parse_main_response(text, thinking=thinking)
        if new_parsed.get("reasoning") and new_parsed.get("answer"):
            return new_parsed["reasoning"], new_parsed["answer"]
        # Fall back to the stored parsed field in case response.text is missing
        stored = r.get("parsed", {}) or {}
        if stored.get("reasoning") and stored.get("answer"):
            return stored["reasoning"], stored["answer"]
    return "", None
