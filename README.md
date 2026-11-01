# STRIDE

**S**tepwise **T**est-guided **R**epair and **I**ncremental **D**ependency **E**volution.

STRIDE extends AURA, a test-guided dependency upgrader, past *code barriers*:
releases that install cleanly but break the project's tests. It alternates two agents:

1. **Version agent** (`aura/`, copied unchanged from AURA): upgrades as far as
   versions alone allow.
2. **Code agent** (plug-in: mini-SWE-agent first, then OpenHands, Aider, SWE-agent):
   changes the code to cross one barrier at a time, with test files locked.

A controller (`stride/`) picks the barrier, builds the evidence for the agent,
validates each change against the tests, rolls back on regressions, and hands
control back to the version agent.

- Design: `docs/DESIGN.md`
- Barrier list from the AURA runs: `docs/barriers.json`
- Pilot inputs (AURA final states for 5 projects): `pilot/aura_results/`

`aura/` is kept as-is so results stay comparable with the AURA baseline.
New code goes in `stride/`.

## Setup

```bash
./scripts/setup_stride.sh          # .devenv with AURA's harness + mini-swe-agent, runs the tests
```

On the cluster, AURA's one-time setup must already be in place (`$AURA_SCRATCH`
with uv Pythons, Ollama and `qwen2.5-coder:14b`); STRIDE reuses
`scripts/slurm/cluster_env.sh`.

## Running

```bash
# Which barriers did AURA leave?
.devenv/bin/python -m stride.cli barriers concept-transformer

# The STRIDE loop (cluster): AURA's final state -> code change -> AURA -> ...
sbatch scripts/slurm/stride.sbatch concept-transformer stride --first torch

# Baselines on the same subject
sbatch scripts/slurm/stride.sbatch concept-transformer pin_latest
sbatch scripts/slurm/stride.sbatch concept-transformer all_latest

# Quick local smoke test: one barrier, no AURA rerun
.devenv/bin/python -m stride.cli run koandeaire-backend --rounds 1 \
    --barriers-per-round 1 --skip-aura
```

Results land in `results/<subject>/<arm>__<agent>/`:

| File | What it is |
|---|---|
| `summary.json` | rounds, every attempt, releases behind / at latest before and after |
| `round<N>_<pkg>/repair.json` | one attempt: triage, accepted?, reason, patch size, agent steps and tokens |
| `round<N>_<pkg>/task.md` | the exact task the agent got |
| `round<N>_<pkg>/trajectory.json` | the agent's full transcript |
| `round<N>_<pkg>/patch.diff` | the code change it made |
| `round<N>_aura.*` | the AURA run after an accepted change |
| `patch_total.diff` | all code changes STRIDE kept |

## How one repair round works

1. Pin the barrier at its **smallest breaking release** (e.g. torch 1.11 -> 2.0.0,
   not 2.8); packages AURA moved with it are left for pip to match.
2. Run the tests. If they already pass, no code change is needed.
3. **Triage**: if no project file is in the failing tracebacks (the break is
   inside installed packages), try pinning the implicated transitive package
   to another release first. No code agent.
4. Otherwise lock tests and requirements (read-only), block pip, run the code agent.
5. Accept only if protected files are untouched, no skip markers were added,
   installed packages are unchanged, and there is no new failure against the
   original baseline. Otherwise roll the code back.
