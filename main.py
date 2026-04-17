"""Main CLI entry point.

Usage:
    uv run python main.py poc            # tiny smoke test (3 questions, 3 runs)
    uv run python main.py run            # full run (all 5 experiments)
    uv run python main.py run --exp 1    # run one experiment only
    uv run python main.py run --models gpt-4o-mini glm-4.7-flash
    uv run python main.py aggregate      # (re)aggregate CSVs from raw files
    uv run python main.py figures        # generate figures
    uv run python main.py tables         # generate LaTeX tables
    uv run python main.py status         # show progress across experiments
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Silence Pydantic serializer UserWarnings emitted by LiteLLM's internal
# response copying. They're cosmetic — the actual text content flows through.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

from dotenv import load_dotenv


ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _build_common(poc: bool):
    from src.utils.api_cache import APICache
    from src.utils.config import load_config
    from src.utils.logger import get_logger
    from src.utils.rate_limiter import RateLimiter

    _load_env()
    cfg = load_config(ROOT / "config.yaml", poc=poc)

    storage = cfg["storage"]
    api = cfg["api"]
    cache = APICache(cache_dir=ROOT / storage["cache_dir"])
    rate = RateLimiter(requests_per_minute=int(api.get("rate_limit_rpm", 60)))

    log_file = Path(storage["raw_dir"]) / ("run_poc.log" if poc else "run_full.log")
    logger = get_logger("main", log_file=log_file)

    return cfg, cache, rate, logger


def _load_questions(cfg):
    from src.data_loader import load_medqa

    data_cfg = cfg["data"]
    return load_medqa(
        hf_dataset_id=data_cfg["hf_dataset_id"],
        hf_subset=data_cfg.get("hf_subset", "med_qa_en_source"),
        split=data_cfg["split"],
        n_questions=int(data_cfg["n_questions"]),
        seed=int(data_cfg["seed"]),
        output_path=ROOT / data_cfg["processed_path"],
        pool_path=ROOT / data_cfg["pool_path"],
    )


def cmd_poc(args):
    cfg, cache, rate, logger = _build_common(poc=True)
    from src.aggregator import save_results_csv
    from src.experiments.exp1_esi import run_exp1
    from src.experiments.exp2_ect import run_exp2
    from src.experiments.exp3_pss import run_exp3
    from src.experiments.exp4_counterfactual import run_exp4
    from src.experiments.exp5_demographic_bias import run_exp5
    from src.llm_client import build_clients

    only_models = args.models or None
    clients = build_clients(cfg, cache=cache, rate_limiter=rate, only=only_models)
    # helper_clients always has ALL models from config, regardless of --models filter.
    # Used for judge/extraction/paraphrase lookups so they always route to the correct
    # provider (e.g. gpt-4o-mini via OpenRouter), never fall back to the model under test.
    helper_clients = build_clients(cfg, cache=cache, rate_limiter=rate, only=None)
    if not clients:
        logger.error("No clients built — check --models flag or config")
        sys.exit(1)
    logger.info("POC: loaded %d clients: %s", len(clients), list(clients.keys()))
    logger.info("POC: helper clients: %s", list(helper_clients.keys()))

    questions = _load_questions(cfg)
    logger.info("POC: %d questions", len(questions))

    agg_dir = ROOT / cfg["storage"]["aggregated_dir"]

    logger.info("=== POC Exp 1 (ESI) ===")
    res = run_exp1(cfg, clients, questions)
    save_results_csv(res, agg_dir / "exp1_esi_results_poc.csv")

    logger.info("=== POC Exp 2 (ECT) ===")
    res = run_exp2(cfg, clients, questions, helper_clients=helper_clients)
    save_results_csv(res, agg_dir / "exp2_ect_results_poc.csv")

    logger.info("=== POC Exp 3 (PSS) ===")
    res = run_exp3(cfg, clients, questions, helper_clients=helper_clients)
    save_results_csv(res, agg_dir / "exp3_pss_results_poc.csv")

    logger.info("=== POC Exp 4 (Counterfactual) ===")
    res = run_exp4(cfg, clients, questions, helper_clients=helper_clients)
    save_results_csv(res, agg_dir / "exp4_counterfactual_results_poc.csv")

    logger.info("=== POC Exp 5 (Demographic Bias) ===")
    res = run_exp5(cfg, clients, questions, helper_clients=helper_clients)
    save_results_csv(res, agg_dir / "exp5_bias_results_poc.csv")

    logger.info("POC COMPLETE. Check results/aggregated/*_poc.csv")


def cmd_run(args):
    cfg, cache, rate, logger = _build_common(poc=False)
    from src.aggregator import save_results_csv
    from src.experiments.exp1_esi import run_exp1
    from src.experiments.exp2_ect import run_exp2
    from src.experiments.exp3_pss import run_exp3
    from src.experiments.exp4_counterfactual import run_exp4
    from src.experiments.exp5_demographic_bias import run_exp5
    from src.llm_client import build_clients

    only_models = args.models or None
    clients = build_clients(cfg, cache=cache, rate_limiter=rate, only=only_models)
    # helper_clients always has ALL models from config regardless of --models filter.
    # Critical: ensures the judge/extraction/paraphrase models always route to their
    # configured provider (e.g. gpt-4o-mini via OpenRouter), never fall back to the
    # model under test (which could have a small context like MedGemma's 4096).
    helper_clients = build_clients(cfg, cache=cache, rate_limiter=rate, only=None)
    if not clients:
        logger.error("No clients built")
        sys.exit(1)
    logger.info("FULL: %d clients (under test): %s", len(clients), list(clients.keys()))
    logger.info("FULL: %d helper clients: %s", len(helper_clients), list(helper_clients.keys()))

    questions = _load_questions(cfg)
    logger.info("FULL: %d questions", len(questions))

    agg_dir = ROOT / cfg["storage"]["aggregated_dir"]
    want_exps = set(args.exp) if args.exp else {1, 2, 3, 4, 5}

    if 1 in want_exps:
        logger.info("=== Exp 1 (ESI) ===")
        res = run_exp1(cfg, clients, questions)
        save_results_csv(res, agg_dir / "exp1_esi_results.csv")

    if 2 in want_exps:
        logger.info("=== Exp 2 (ECT) ===")
        res = run_exp2(cfg, clients, questions, helper_clients=helper_clients)
        save_results_csv(res, agg_dir / "exp2_ect_results.csv")

    if 3 in want_exps:
        logger.info("=== Exp 3 (PSS) ===")
        res = run_exp3(cfg, clients, questions, helper_clients=helper_clients)
        save_results_csv(res, agg_dir / "exp3_pss_results.csv")

    if 4 in want_exps:
        logger.info("=== Exp 4 (Counterfactual) ===")
        res = run_exp4(cfg, clients, questions, helper_clients=helper_clients)
        save_results_csv(res, agg_dir / "exp4_counterfactual_results.csv")

    if 5 in want_exps:
        logger.info("=== Exp 5 (Demographic Bias) ===")
        res = run_exp5(cfg, clients, questions, helper_clients=helper_clients)
        save_results_csv(res, agg_dir / "exp5_bias_results.csv")

    logger.info("FULL RUN COMPLETE.")


def cmd_aggregate(args):
    cfg, _, _, logger = _build_common(poc=args.poc)
    from src.aggregator import summarize_by_model

    agg_dir = ROOT / cfg["storage"]["aggregated_dir"]
    suffix = "_poc" if args.poc else ""
    files = [
        ("exp1", f"exp1_esi_results{suffix}.csv", ["esi_score", "answer_consistency", "key_factor_jaccard", "correct_rate"]),
        ("exp2", f"exp2_ect_results{suffix}.csv", ["cfs_score", "n_concepts", "n_causal"]),
        ("exp3", f"exp3_pss_results{suffix}.csv", ["answer_stability", "explanation_stability", "concept_stability", "confidence_drift"]),
        ("exp4", f"exp4_counterfactual_results{suffix}.csv", ["counterfactual_valid", "minimality", "plausibility"]),
        ("exp5", f"exp5_bias_results{suffix}.csv", ["edd_score", "n_stereotype_flags"]),
    ]
    for exp, fname, cols in files:
        path = agg_dir / fname
        if not path.exists():
            logger.warning("Missing %s", path)
            continue
        summary = summarize_by_model(path, cols)
        logger.info("\n=== %s summary ===\n%s", exp, summary.to_string())


def cmd_figures(args):
    from scripts.generate_figures import main as generate_figures
    generate_figures(poc=args.poc)


def cmd_tables(args):
    from scripts.generate_tables import main as generate_tables
    generate_tables(poc=args.poc)


def cmd_status(args):
    cfg, _, _, logger = _build_common(poc=False)
    raw_dir = ROOT / cfg["storage"]["raw_dir"]
    for exp in ["exp1_esi", "exp2_ect", "exp3_pss", "exp4_counterfactual", "exp5_bias"]:
        exp_dir = raw_dir / exp
        if not exp_dir.exists():
            logger.info("%s: (not started)", exp)
            continue
        for model_dir in exp_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith("_"):
                count = len(list(model_dir.glob("*.json")))
                logger.info("%s / %s: %d files", exp, model_dir.name, count)


def main():
    parser = argparse.ArgumentParser(prog="counterfactual-clinical-xai")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_poc = sub.add_parser("poc", help="Run a tiny POC smoke test")
    p_poc.add_argument("--models", nargs="*", help="Restrict to these model names")
    p_poc.set_defaults(func=cmd_poc)

    p_run = sub.add_parser("run", help="Full experiment run")
    p_run.add_argument("--models", nargs="*", help="Restrict to these model names")
    p_run.add_argument("--exp", type=int, nargs="*", choices=[1, 2, 3, 4, 5], help="Only run listed experiments")
    p_run.set_defaults(func=cmd_run)

    p_agg = sub.add_parser("aggregate", help="Print per-model summary tables")
    p_agg.add_argument("--poc", action="store_true")
    p_agg.set_defaults(func=cmd_aggregate)

    p_fig = sub.add_parser("figures", help="Generate paper figures")
    p_fig.add_argument("--poc", action="store_true")
    p_fig.set_defaults(func=cmd_figures)

    p_tab = sub.add_parser("tables", help="Generate LaTeX tables")
    p_tab.add_argument("--poc", action="store_true")
    p_tab.set_defaults(func=cmd_tables)

    p_stat = sub.add_parser("status", help="Show progress")
    p_stat.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
