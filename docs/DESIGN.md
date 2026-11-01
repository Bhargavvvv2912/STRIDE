# AURA-X: alternating version search and code repair

Status: week 1 draft, 2026-11-01. Target: ASE 2027 or ICSE 2028.

## 1. Problem

AURA keeps the code fixed and searches versions. It stops at a **code barrier**:
a release that installs cleanly but breaks tests because the API changed
(for example torch 1 to 2, SQLAlchemy 1.4 to 2.0, Django 3 to 4). Code-repair
agents do the opposite: they fix the target version and edit the code, which
forces them to absorb every breaking change at once.

AURA-X searches over **(code state, version state)** together: version search
up to a barrier, a small code change across that one barrier, then version
search again.

## 2. Evidence that the problem is common (from our own logs)

`scripts/mine_barriers.py` (headline arm `aura_full_qwen14b`) finds, among the
50 SEAMS projects, **93 code barriers in 30 projects, 75 at a major version**.
The full list is in `barriers.json`. The most frequent barrier packages are
flask (12), werkzeug (8), sqlalchemy (7), flask-sqlalchemy (6) and pytest (4).
A barrier here means a newer release of a package that is not at latest
installed, but new tests failed. The API-error flag is a loose regex, so check
it by hand.

## 3. Prior work and the gap

| Work | Searches versions | Changes code | Notes |
|---|---|---|---|
| AURA (ours, SEAMS 2027 sub.) | yes | no | test-guided, tiered |
| DepUpdater (ICSE 26), PCREQ | yes | no | Java / Python, static or constraint-based |
| DepUpgrade (ASE 26) | no (v_old to v_new given) | yes | searches code snapshots only; Java; compiler feedback; N=5 rounds; rollback with a "progressive state" rule; API-diff knowledge base. Lists dynamic languages as untested. Borrow: rollback rule, file-level tasks, API diff |
| LADU / J.P. Morgan (ASE-W 25) | no (target given) | yes | 3 agents, Java |
| DepBench (2608.30300) | no | yes | 203 tasks; best agent 51%; main failure = incomplete migration |
| RepoRescue (2607.01213) | no (harness *forbids* dependency edits and pip install) | yes | Says banning dependency changes "abstracts away a channel real maintainers use (29.8% of time-travel fixes)". Claude Code successes rely on test edits 38-53% of the time; agents lose their best state in 30% of failed runs; union beats the best agent by ~11 pp. Borrow: runtime test-write blocking, pass count within 5% of baseline |
| SWE-Chain (2605.14415) | no | yes | chained releases, but of the *library itself*, steps fixed |
| BBCFixer, DepRepair, Byam, BigBag | no | yes | repair after a fixed bump |
| Routing study (2609.25911) | no | routes | decides whether to call a repair agent |

**Gap:** no published system lets the search choose versions *and* code, with
the tests as the oracle. Two findings support stepping one barrier at a time:
DepBench (agents fail on large migrations) and SWE-Chain (agents lose
earlier changes across a chain).

## 4. Architecture

```
          +------------------- controller (ours) -------------------+
          |                                                          |
 project -> AURA version search -> barrier selector -> evidence pack  |
          ^                                              |           |
          |                                     code agent (plug-in) |
          |                                              |           |
          +---- accept if tests pass <- validator <------+           |
          +----------------------------------------------------------+
```

1. **Version agent**: AURA as published, run to a fixed point.
2. **Barrier selector**: picks one barrier `(package, target release)`.
   - Policies to compare: most blocking (unlocks the most others), smallest
     jump first, or oldest first.
   - The target is the **smallest breaking release**, not latest. This keeps
     each code change small.
3. **Evidence pack** for the code agent:
   - the new failing tests and their traces
   - the API diff or changelog between the two releases
   - the call sites found by import scan
   - the pinned environment at the target release
4. **Code agent (plug-in)**: mini-SWE-agent, OpenHands, Aider, Claude Code or
   Codex. It gets one barrier per call and a fixed step and token budget.
5. **Validator**:
   - Test files are read-only and hash-checked. The run is rejected if the
     agent skipped or deleted a test, or added a new `xfail`.
   - Baseline-relative check, as in AURA.
6. **Loop**: after an accepted patch, rerun the version search from the new
   code. Stop when no barriers remain or the budget runs out.
7. **Rollback**: if a patch blocks a later upgrade, or the next round
   regresses, revert to the last good (code, versions) state.

Knowledge (MAPE-K): accepted patches per (package, version pair) become
reusable hints for later projects. This links to BigBag and to our KTO work.

## 5. Research questions (draft)

- RQ1: Does alternation reach more packages at latest than AURA alone and than
  latest-then-repair, with tests still passing?
- RQ2: Does stepping one barrier at a time give smaller, more correct patches
  than a single jump to latest?
- RQ3: Does the gain hold across code agents (2 to 3 agents, open and closed)?
- RQ4: Which controller choices matter? Ablations: barrier order, jump size,
  evidence pack, rollback.
- RQ5: Cost: agent tokens, install time, number of rounds.

## 6. Arms

1. AURA only (SEAMS result, reused)
2. Code agent only, target = latest
3. Latest then repair: pin latest, give the agent all failures (DepBench style)
4. AURA-X (alternating)

Plus ablations of arm 4.

## 7. Metrics

- At latest (%) and releases behind (as in SEAMS)
- Tests: no new failures; no test edits or skips
- Patch size (files, lines)
- Manual correctness review of a sample
- Cost (tokens, wall clock)

## 8. Pilot (weeks 2 to 3)

Five projects with different stacks (not only Flask):

| Project | Barrier |
|---|---|
| concept-transformer | torch 1.11 to 2.x, pytorch-lightning 1.6 to 2.x, numpy 1 to 2 |
| koandeaire-backend | fastapi 0.76 to 0.100+ (pydantic 2), sqlalchemy 1.4 to 2 |
| authors-of-leisure | django 3.0 to 4.2 |
| find-that-postcode | elasticsearch 7 to 9 |
| graphene-mongodb | graphql-core 2 to 3, graphene 2 to 3 |

Procedure:
1. Start from the AURA final state.
2. Run arm 3 and a hand-driven arm 4, both with mini-SWE-agent and the same
   budget.
3. Compare packages crossed, tests, patch size and cost.

Go/no-go rule: arm 4 beats arm 3 on reach or on patch size/correctness in at
least 3 of 5 projects.

## 9. Risks

- **Test gaming**: RepoRescue saw agents edit tests. Use locked files and a
  skip/xfail check.
- **Weak tests**: a patch can pass and still be wrong. Use manual review and
  report patch size.
- **Leakage**: models may have seen these migrations. Include barriers released
  after the model's cutoff.
- **Cost**: agent runs dominate. Cap the budget per barrier and reuse AURA's
  batching.
- **Scope creep**: Python only, pip only.

## 10. Week 1 remaining

- [x] Read DepUpgrade in full: versions never vary.
- [x] Read RepoRescue: dependency edits are forbidden by the harness.
- [ ] Check the barrier list by hand for the 5 pilot projects.
- [x] Code agent for the pilot: mini-SWE-agent. Later: OpenHands, Aider, SWE-agent (all run on local Qwen via Ollama); optionally OpenCode.
