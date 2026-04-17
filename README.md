# Counterfactual Clinical XAI — Experiment Orchestration

Paper: **"Are Clinical LLM Explanations Reliable? Measuring Stability, Causal Faithfulness, and Robustness of Medical Chain-of-Thought Reasoning"**

Five experiments on MedQA USMLE questions across 6 LLMs:

| # | Role | Model | Served via |
|---|---|---|---|
| 1 | Regular | `openai/gpt-4o-mini` | OpenRouter |
| 2 | Regular | `z-ai/glm-4.7-flash` | OpenRouter |
| 3 | Reasoning | `deepseek/deepseek-r1-distill-qwen-32b` | OpenRouter |
| 4 | Reasoning | `minimax/minimax-m2.5` (thinking ON) | OpenRouter |
| 5 | Medical | `unsloth/medgemma-27b-text-it-GGUF` @ Q4_K_M | LM Studio local |
| 6 | Medical | `QuantFactory/OpenBioLLM-Llama3-8B-GGUF` @ Q8_0 | LM Studio local |

Five experiments:
1. **ESI** — Explanation Stability Index (10 reruns per Q, pairwise cosine similarity)
2. **ECT** — Explanation Causality Test (concept ablation → causal faithfulness score)
3. **PSS** — Perturbation Stability Score (paraphrase the question, check stability)
4. **Counterfactual Validity** — model generates CF, we verify it actually flips the answer
5. **Demographic Bias** — 7 demographic variants, measure explanation divergence

---

## 0. One-time setup

```bash
# 0.1 Install deps
uv sync

# 0.2 Copy env template and fill in your OpenRouter key
cp .env.example .env
# Edit .env → paste your OPENROUTER_API_KEY
```

Get an OpenRouter key at https://openrouter.ai/keys (pay-as-you-go; ~$5 should cover the entire paper).

### Install LM Studio and download the 2 medical models

1. Download LM Studio: https://lmstudio.ai/
2. Launch it. Use the in-app **Discover / Search** tab to download:
   - `unsloth/medgemma-27b-text-it-GGUF` → pick `Q4_K_M` (~16.5 GB)
   - `QuantFactory/OpenBioLLM-Llama3-8B-GGUF` → pick `Q8_0` (~8.5 GB)
3. Go to the **Developer** tab → click **Start Server** (it listens on `http://localhost:1234/v1`).
4. Load **one** of the two models into memory with "GPU Offload: max". Run the medical experiments against that model; when done, unload it and load the other one.

> 24 GB VRAM is too tight to hold both models in memory simultaneously. Run them sequentially.

### Accept MedGemma license (one-time)

Go to https://huggingface.co/google/medgemma-27b-text-it while logged into HuggingFace and accept the Health AI Developer Foundations terms. This is required to access the model weights (needed only once).

---

## 1. POC (proof-of-concept smoke test)

**ALWAYS run this first** before the full run. It uses 3 questions, 3 reruns, 2 counterfactuals, 2 bias questions — completes in a few minutes and costs ~$0.05.

### 1a. POC without the medical models (fastest — OpenRouter only)

```bash
uv run python main.py poc --models gpt-4o-mini glm-4.7-flash deepseek-r1-distill-qwen-32b minimax-m2.5
```

This verifies:
- `.env` and OpenRouter key work
- Parsers handle each model's output format
- Cache hits/misses behave correctly (try re-running — should say `cached=True`)
- All 5 experiments run end-to-end
- Results land in `results/raw/exp*/` and `results/aggregated/*_poc.csv`

### 1b. POC with the medical models (after LM Studio is set up)

Load **MedGemma-27B** in LM Studio → Start Server → run:

```bash
uv run python main.py poc --models medgemma-27b
```

Then unload MedGemma → load **OpenBioLLM-8B** → Start Server → run:

```bash
uv run python main.py poc --models openbiollm-8b
```

### 1c. Inspect POC results

```bash
uv run python main.py aggregate --poc      # print per-model summaries
uv run python main.py figures --poc        # generate figures from POC data
uv run python main.py tables --poc         # generate LaTeX tables
uv run python main.py status               # show how many raw json files exist per exp/model
```

POC outputs:
- `results/raw/exp1_esi/<model>/<question_id>.json` — full raw responses per question
- `results/aggregated/exp*_results_poc.csv` — per-question metric rows
- `results/figures/figure*.pdf|png` — generated from POC data
- `paper/tables/table*_poc.tex` — LaTeX tables

If everything looks sane, move on.

---

## 2. Full run

### 2a. OpenRouter models (4 models × ~9,000 calls ≈ 1–2 hours, ~$10)

```bash
uv run python main.py run --models gpt-4o-mini glm-4.7-flash deepseek-r1-distill-qwen-32b minimax-m2.5
```

You can also run one experiment at a time:

```bash
uv run python main.py run --exp 1                        # all 6 models, exp 1 only
uv run python main.py run --exp 1 2 --models gpt-4o-mini # just gpt-4o-mini, exps 1 and 2
```

Experiments 2 and 4 depend on the output of exp 1 (they reuse the model's first-run reasoning), so **always run exp 1 first** for a given model.

### 2b. Medical models (sequential, ~17 hours each unattended)

Load MedGemma-27B in LM Studio → Start Server:

```bash
uv run python main.py run --models medgemma-27b
```

Unload MedGemma → load OpenBioLLM-8B → Start Server:

```bash
uv run python main.py run --models openbiollm-8b
```

### 2c. Aggregate + figures + tables

```bash
uv run python main.py aggregate
uv run python main.py figures
uv run python main.py tables
```

Outputs:
- `results/figures/figure1.pdf ... figure7.pdf` (the Reliability Quadrant is fig 7)
- `paper/tables/table1_main.tex`, `table2_cited_unused.tex`, `table3_counterfactual_examples.tex`

---

## 3. Resume / re-run semantics

Everything is cached via `diskcache` in `results/raw/.cache/`. If a run crashes, just re-run the same command — cached API responses return instantly at $0. Per-question JSON files in `results/raw/exp*/<model>/` are also incremental: existing runs are loaded and topped up.

To **force a clean re-run** of a specific question, delete its JSON file from `results/raw/exp*/<model>/<question_id>.json`. To clear the API cache entirely, delete `results/raw/.cache/`.

---

## 4. Common commands

| Command | Purpose |
|---|---|
| `uv run python main.py poc` | POC smoke test (3 questions × all models) |
| `uv run python main.py poc --models gpt-4o-mini` | POC for one model only |
| `uv run python main.py run` | Full run, all 5 experiments, all 6 models |
| `uv run python main.py run --exp 1` | Only exp 1 |
| `uv run python main.py run --models glm-4.7-flash` | Only one model |
| `uv run python main.py aggregate` | Print metric summary tables |
| `uv run python main.py figures` | Generate PDF+PNG figures |
| `uv run python main.py tables` | Generate LaTeX tables |
| `uv run python main.py status` | Show how much has been run |

Add `--poc` to `aggregate`, `figures`, `tables` to work on POC outputs.

---

## 5. File layout

```
counterfactual-clinical-xai/
├── config.yaml              # all experiment parameters + models
├── .env                     # API keys (gitignored)
├── .env.example
├── main.py                  # CLI entry point
├── pyproject.toml           # uv-managed pinned deps
├── data/
│   └── processed/medqa_100.jsonl
├── src/
│   ├── llm_client.py        # unified LiteLLM client (OpenRouter + LM Studio)
│   ├── prompts.py           # ALL prompts + response parsers
│   ├── data_loader.py       # MedQA loader + demographic variant generator
│   ├── aggregator.py
│   ├── utils/
│   │   ├── api_cache.py     # diskcache wrapper
│   │   ├── config.py        # yaml loader with POC override
│   │   ├── logger.py
│   │   └── rate_limiter.py
│   ├── metrics/
│   │   ├── esi.py
│   │   ├── ect.py
│   │   ├── pss.py
│   │   ├── counterfactual.py
│   │   └── similarity.py    # sentence-transformers wrapper
│   └── experiments/
│       ├── common.py
│       ├── exp1_esi.py
│       ├── exp2_ect.py
│       ├── exp3_pss.py
│       ├── exp4_counterfactual.py
│       └── exp5_demographic_bias.py
├── scripts/
│   ├── generate_figures.py
│   └── generate_tables.py
├── results/
│   ├── raw/
│   │   ├── .cache/                         # diskcache files
│   │   ├── exp1_esi/<model>/<qid>.json     # all 10 runs per question
│   │   ├── exp2_ect/<model>/<qid>.json     # concepts + ablations + cfs
│   │   ├── exp3_pss/<model>/<qid>.json
│   │   ├── exp3_pss/_paraphrases/<qid>.json  # shared across models
│   │   ├── exp4_counterfactual/<model>/<qid>.json
│   │   └── exp5_bias/<model>/<qid>.json
│   ├── aggregated/
│   │   ├── exp1_esi_results.csv
│   │   ├── exp2_ect_results.csv
│   │   ├── exp3_pss_results.csv
│   │   ├── exp4_counterfactual_results.csv
│   │   └── exp5_bias_results.csv
│   └── figures/
└── paper/
    └── tables/
```

---

## 6. Cost estimate

| Experiment | Calls/model | OpenRouter cost/model | LM Studio cost |
|---|---|---|---|
| Exp 1 (ESI) | ~1000 | ~$0.60 | $0 |
| Exp 2 (ECT) | ~500 | ~$0.30 | $0 |
| Exp 3 (PSS) | ~400 | ~$0.25 | $0 |
| Exp 4 (CF) | ~320 | ~$0.45 | $0 |
| Exp 5 (Bias) | ~420 | ~$0.30 | $0 |
| **Total per OR model** | **~2640** | **~$2** | — |

4 OpenRouter models × ~$2 = **~$8 total API cost**. Medical models run free on your GPU.

POC run is **under $0.10** total.

---

## 7. Troubleshooting

- **`OpenRouter: authentication failed`** → check `.env` has a valid `OPENROUTER_API_KEY`.
- **`Connection refused: localhost:1234`** → LM Studio server not started. Open LM Studio → Developer tab → Start Server.
- **`Model not loaded`** (LM Studio) → load the GGUF into memory before running; LM Studio won't auto-load.
- **MedQA download fails** → set `HF_TOKEN` in `.env` and/or try the fallback dataset id `openlifescienceai/medqa` (edit `config.yaml` → `data.hf_dataset_id`).
- **`CUDA out of memory`** → lower LM Studio's GPU-offload slider or pick a smaller quant (Q3_K_M instead of Q4_K_M).
- **Parser shows `parse_ok: false`** → the model isn't following the REASONING:/ANSWER: format. Inspect the raw JSON in `results/raw/exp1_esi/<model>/<qid>.json`. Usually harmless for a few edge cases.

---

## 8. Credits

- MedQA: https://huggingface.co/datasets/bigbio/med_qa
- MedGemma: Google Health AI Developer Foundations
- OpenBioLLM: Saama AI
- Built on LiteLLM, diskcache, sentence-transformers, matplotlib, seaborn
