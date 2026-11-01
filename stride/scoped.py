"""Scoped version search: after a code change, search only what it can affect.

A full AURA rerun searches every dependency again (about 1.5 h on
concept-transformer), although a code change across one barrier can only
unblock the crossed package and packages coupled to it. The scoped search
starts from those, and widens itself as it goes: whenever a package moves,
its coupled neighbours (from AURA's own constraint graph) join the scope.

``aura/`` is not modified. A subclass filters AURA's candidate list, and is
swapped into AURA's own runner only for the length of one STRIDE rerun, so
logging, budgets and results are exactly AURA's.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

from aura.agent import AuraAgent
from aura.telemetry import Event


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def make_scoped_agent(seed: set[str], hops: int = 1):
    """An AuraAgent class whose search covers ``seed`` plus coupled packages."""
    seed = {_norm(s) for s in seed}

    class ScopedAuraAgent(AuraAgent):
        scope: set[str] = set()

        def _neighbours(self, names: set[str]) -> set[str]:
            graph = getattr(self, "graph", None)
            if graph is None:
                return set()
            out = set()
            for pair in graph.couples():
                a, b = (_norm(x) for x in pair)
                if a in names:
                    out.add(b)
                if b in names:
                    out.add(a)
            return out

        def build_candidates(self, reqfile, spread, statements):
            if not self.scope:
                self.scope = set(seed)
                self._start_pins = {_norm(k): v for k, v in reqfile.pins().items()}
            # Packages that moved since the rerun began unlock their neighbours.
            moved = {_norm(k) for k, v in reqfile.pins().items()
                     if self._start_pins.get(_norm(k)) != v}
            frontier = self.scope | moved
            for _ in range(hops):
                frontier |= self._neighbours(frontier)
            self.scope = frontier
            everything = super().build_candidates(reqfile, spread, statements)
            kept = [c for c in everything if _norm(c.name) in self.scope]
            self.log.emit(Event.NOTE, note="stride_scope", scope=sorted(self.scope),
                          kept=[c.name for c in kept],
                          skipped=sorted(c.name for c in everything if c not in kept))
            self.log.bump("stride_scope_skipped", len(everything) - len(kept))
            return kept

    return ScopedAuraAgent


@contextlib.contextmanager
def scoped_search(seed: set[str], hops: int = 1):
    """Within this block, evaluation.runner.run_aura_arm uses the scoped agent."""
    import evaluation.runner as runner
    original = runner.AuraAgent
    runner.AuraAgent = make_scoped_agent(seed, hops)
    try:
        yield
    finally:
        runner.AuraAgent = original


def seed_for(crossed_package: str, plan: dict[str, str],
             extra: dict[str, str] | None = None) -> set[str]:
    """What a crossed barrier can unblock: the package, the packages AURA
    moved with it, and any transitive pins STRIDE added."""
    seed = {crossed_package} | set(plan)
    seed |= set(extra or {})
    return {_norm(s) for s in seed}


def carry_forward(previous: list, fresh: list, pins: dict[str, str]) -> list:
    """Barriers for the next round. A scoped rerun only logs the packages it
    searched, so a barrier outside its scope is still valid while that
    package's pin is unchanged; fresh barriers win over old ones."""
    current = {_norm(k): v for k, v in pins.items()}
    have = {_norm(b.package) for b in fresh}
    kept = [b for b in previous if _norm(b.package) not in have
            and current.get(_norm(b.package)) == b.current]
    return fresh + kept
