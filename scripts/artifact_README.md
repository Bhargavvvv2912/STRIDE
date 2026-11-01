# AURA: replication package

This package contains AURA's source code, the 50 projects of the study, every
run of every approach on them, and the benchmark and data behind each result
in the paper. The tool's own usage guide is in `TOOL_README.md`.

## Layout

| Path | What it holds |
|---|---|
| `aura/` | AURA itself: the three tiers, runtime verification, the LLM step |
| `evaluation/` | The study harness: one *arm* per approach, run on one project at a time |
| `subjects.json`, `DATASET.md` | The 50 projects: repository URL, pinned commit, Python version, how each is tested |
| `baselines/` | Hand-fixed baselines for the few projects whose rebuild needed one |
| `screening/` | The subject-selection records (Section 4.1): every candidate and why it was kept or rejected |
| `results/<project>/` | Every approach's run on that project (see below) |
| `results/*.json` | Benchmark, replay and prompt-design results (RQ5) |
| `data/bench/` | The 407-case coupled-upgrade benchmark (RQ5) |
| `data/ft/` | Fine-tuning data (RQ5): every LLM reply from the study's runs, labelled by what happened when AURA tried it (`kto.jsonl`), and the accepted ones alone (`sft.jsonl`) |
| `analysis/` | Per-project numbers behind the RQ2 figure |
| `scripts/` | Running, collecting and analysis scripts |

## One run, one set of files

For each project and approach (`<arm>`), `results/<project>/` holds:

- `<project>__<arm>.result.json`: the outcome, the upgraded packages, and the
  share of packages at their latest version before and after;
- `<project>__<arm>.jsonl`: every event of the run (each install, test run,
  LLM call with its prompt and reply, and why each candidate was accepted or
  rejected);
- `<project>__<arm>.log`: the same run, readable;
- `<project>__<arm>.requirements.txt`: the final versions.

| Approach in the paper | `<arm>` |
|---|---|
| pip upgrade | `pip_upgrade` |
| Dependabot | `dependabot` (plus `.github.json`: every pull request and its CI result) |
| Renovate | `renovate_replay` (plus `renovate.proposals.json`: every update Renovate proposed) |
| AURA without the LLM | `aura_linear_greedy` |
| AURA | `aura_full_<model>` (`qwen3b`, `llama8b`, `qwen14b`, `qwen32b`) |
| LLM-only | `aura_llm_<model>` |
| One-pass | `llm_oneshot_<model>` |
| Baseline (the frozen starting point) | `baseline` |

The paper's AURA results use `aura_full_qwen14b`.

## Reproducing the results

Requirements: Python 3.11 for the harness (`pip install -r requirements-harness.txt`),
the Python versions the projects target (installed automatically through `uv`),
and, for the LLM approaches, [Ollama](https://ollama.com) with the models above.

**Run an approach on a project** (RQ1 to RQ4):

```
python -m evaluation.cli run --registry subjects.json \
    --arms aura_full_qwen14b --subjects concept-transformer
python -m evaluation.cli arms        # every approach and its arm name
```

Results are written to `results/<project>/` as described above.

**Renovate.** Renovate's proposals come from the Renovate CLI (version 44, run
in Docker, offline) with the configuration in `scripts/real_renovate.py`:

```
python scripts/renovate_proposals.py <project>
python -m evaluation.cli run --registry subjects.json --arms renovate_replay --subjects <project>
```

**Dependabot** runs on GitHub (`scripts/real_dependabot.py setup|collect <project>`);
its collected pull requests are included in `results/`.

**Tables and figures** (RQ1 to RQ4): `python scripts/build_report.py` reads every
`result.json` and writes the per-project numbers to `results/report_data.json`.

**Benchmark and fine-tuning** (RQ5):

```
python scripts/bench/build_cases.py ...            # how data/bench/ was built
python scripts/bench/eval_cases.py --cases data/bench/cases_0.jsonl \
    --models qwen2.5-coder:14b --out results/bench_eval.json
python scripts/ft/train_kto.py ...                 # KTO fine-tuning
python scripts/ft/train_qlora.py ...               # supervised fine-tuning
python scripts/ft/replay_eval.py ...               # valid plans on real prompts
```

Each script's header documents its options.

## Note on dates

"Latest version" is measured against PyPI as it stood on one fixed date
(Section 4.1). PyPI keeps changing, so a rerun today proposes newer releases;
the logs record exactly which versions each run saw.
