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
