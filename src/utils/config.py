"""Config loader with POC mode override."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml", poc: bool = False) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path.resolve()}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if poc or cfg.get("mode") == "poc":
        cfg["mode"] = "poc"
        poc_cfg = cfg.get("poc", {})
        # Override sample counts in experiment configs
        cfg["data"]["n_questions"] = poc_cfg.get("n_questions", 3)
        cfg["exp1_esi"]["n_runs"] = poc_cfg.get("n_runs", 3)
        cfg["exp3_pss"]["n_perturbations"] = min(cfg["exp3_pss"].get("n_perturbations", 3), 2)
        cfg["exp4_counterfactual"]["n_questions"] = poc_cfg.get("n_counterfactual", 2)
        cfg["exp5_bias"]["n_questions"] = poc_cfg.get("n_bias", 2)

    return cfg


def get_model_by_name(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    raise KeyError(f"Model '{name}' not found in config")
