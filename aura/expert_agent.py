# expert_agent.py

import re
import json
from google.api_core.exceptions import ResourceExhausted
from packaging.version import parse as parse_version
from typing import Union, Optional, List, Dict

class ExpertAgent:
    """
    The "Expert" Agent (CORE). 
    A Neuro-Symbolic reasoning engine designed for dependency constraint optimization.
    """
    def __init__(self, llm_client):
        self.llm = llm_client
        self.llm_available = True

    def _clean_json_response(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
        return cleaned.strip()

    def _extract_key_constraints(self, error_log: str) -> list:
        key_lines = []
        patterns = [
            r"^\s*([a-zA-Z0-9\-_]+.* requires .*)$",
            r"^\s*([a-zA-Z0-9\-_]+.* depends on .*)$",
            r"^\s*(The user requested .*)$",
            r"^\s*(Incompatible versions: .*)$",
            r"^\s*(Conflict: .*)$"
        ]
        for pat in patterns:
            for match in re.finditer(pat, error_log, re.MULTILINE):
                key_lines.append(match.group(1).strip())
        return list(set(key_lines))[:15]

    def summarize_error(self, error_message: str) -> str:
        if not self.llm_available: return "(LLM summary unavailable)"
        key_constraints = self._extract_key_constraints(error_message)
        context = "\n".join(key_constraints) if key_constraints else error_message[:2500]
        prompt = (
            "Summarize the root cause of the dependency conflict in one sentence. "
            "CRITICAL: You MUST include the specific package versions mentioned in the log. "
            f"Context: {context}"
        )
        try:
            return self.llm.generate_content(prompt).text.strip().replace('\n', ' ')
        except Exception:
            return "Failed to get summary from LLM."

    def diagnose_conflict_from_log(self, error_log: str) -> List[str]:
        found_packages = set()
        pattern_std = re.compile(r"(?P<name>[a-zA-Z0-9\-_]+)(?:==|>=|<=|~=|!=|<|>)")
        pattern_space = re.compile(r"(?P<name>[a-zA-Z0-9\-_]+)\s+\d+(?:\.\d+)+")
        pattern_paren = re.compile(r"(?P<name>[a-zA-Z0-9\-_]+)\s*\(\d+(?:\.\d+)+\)")

        for pat in [pattern_std, pattern_space, pattern_paren]:
            for match in pat.finditer(error_log):
                name = match.group('name').lower()
                if self._is_valid_package_name(name): found_packages.add(name)

        context_keywords = [
            r"conflict(?:s)?\s+(?:between|among|with|in)\s+((?:[a-zA-Z0-9\-_]+(?:,?\s+and\s+|,?\s*)?)+)",
            r"requirement\s+((?:[a-zA-Z0-9\-_]+)+)",
        ]
        for keyword_pat in context_keywords:
            for match in re.finditer(keyword_pat, error_log, re.IGNORECASE):
                raw_list = match.group(1)
                tokens = re.split(r'[,\s]+', raw_list)
                for t in tokens:
                    clean_t = t.strip("`'").lower()
                    if self._is_valid_package_name(clean_t): found_packages.add(clean_t)

        if '-' in found_packages: found_packages.remove('-')
        return list(found_packages)

    def _is_valid_package_name(self, name: str) -> bool:
        noise = {'python', 'pip', 'setuptools', 'wheel', 'setup', 'dependencies', 
                 'versions', 'requirement', 'conflict', 'between', 'and', 'the', 'version', 'package', 'for'}
        return name and len(name) > 1 and name not in noise

    def propose_co_resolution(
        self, target_package: str, error_log: str, available_updates: dict,
        current_versions: dict = None, history: list = None
    ) -> Optional[dict]:
        """
        Iterative Co-Resolution Planner with STRICT FORWARD-ONLY Validation
        and Dynamic Early Exit for environmentally unresolvable conflicts.
        """
        if not self.llm_available: return None

        floor_constraints = json.dumps(current_versions, indent=2) if current_versions else "{}"
        ceiling_constraints = json.dumps(available_updates, indent=2)

        history_text = ""
        if history:
            history_text = "--- PREVIOUS FAILED ATTEMPTS ---\n"
            for i, (attempt_plan, failure_reason) in enumerate(history):
                history_text += f"Attempt {i+1}: {attempt_plan} -> Failed: {failure_reason}\n"

        # ---------------------------------------------------------------
        # CHANGE 1: Updated prompt with resolvable field and early exit rule
        # ---------------------------------------------------------------
        prompt = f"""
        You are CORE (Constraint Optimization & Resolution Expert).
        Solve the dependency deadlock for '{target_package}'.
        The Greedy Heuristic (Max Versions) failed.

        OBJECTIVE:
        Propose a set of versions to update that satisfies all graph constraints.
        
        STRICT RULES:
        1. FORWARD ONLY: **NEVER** propose a version lower than or equal to the CURRENT INSTALLED VERSION.
        2. INTERMEDIATE VERSIONS: You MAY propose versions between Current (Floor) and Available (Ceiling) if the error log suggests it.
        3. SCOPE: The plan MUST include '{target_package}'.
        4. SANITY: Do not invent versions that do not exist.

        EARLY EXIT RULE:
        If you determine that the conflict CANNOT be resolved by any version combination,
        for example because the required Python version is fundamentally incompatible with
        the current runtime, a package requires a system-level compiler that is not present,
        or the dependency chain contains a mathematically impossible constraint that no
        version combination can satisfy, then set "resolvable" to false and explain why
        in the "reason" field. Do NOT propose a plan in this case. This will immediately
        stop further retry attempts and save unnecessary computation.

        CONTEXT:
        Target: {target_package}
        Current (Floor): {floor_constraints}
        Available (Ceiling): {ceiling_constraints}
        
        LOG:
        {error_log}

        {history_text}

        Return JSON only, no markdown:
        {{
          "resolvable": true or false,
          "reason": "only required if resolvable is false — one sentence explanation",
          "plausible": true or false,
          "proposed_plan": ["pkg==ver", ...]
        }}
        """

        try:
            response = self.llm.generate_content(prompt)
            clean_text = self._clean_json_response(response.text)
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if not match: return None
            
            plan = json.loads(match.group(0))

            # ---------------------------------------------------------------
            # CHANGE 2: Check resolvable field BEFORE doing anything else.
            # If the LLM signals the conflict is environmentally unresolvable,
            # return immediately so the calling loop can break early.
            # ---------------------------------------------------------------
            if not plan.get("resolvable", True):
                reason = plan.get("reason", "LLM determined conflict is unresolvable.")
                print(f"  -> EARLY EXIT: {reason}")
                # Return a sentinel that agent_logic.py can detect
                return {"resolvable": False, "reason": reason, "proposed_plan": []}
            
            if plan.get("plausible") and isinstance(plan.get("proposed_plan"), list):
                valid_plan = []
                for requirement in plan.get("proposed_plan", []):
                    try:
                        pkg, ver = requirement.split('==')
                        
                        # --- STRICT VALIDATION LOGIC ---
                        curr_ver_str = current_versions.get(pkg, "0.0.0") if current_versions else "0.0.0"
                        latest_ver_str = available_updates.get(pkg, "9999.9.9")
                        
                        try:
                            v_prop = parse_version(ver)
                            v_curr = parse_version(curr_ver_str)
                            v_max = parse_version(latest_ver_str)
                            
                            if v_prop > v_curr:
                                valid_plan.append(requirement)
                            else:
                                print(f"  -> LLM_WARNING: Filtered version {pkg}=={ver} (Not an upgrade from {curr_ver_str})")
                        except:
                            pass

                    except ValueError: continue
                
                if not valid_plan:
                    print("  -> LLM_WARNING: Plan filtered because it contained no upgrades.")
                    return {"resolvable": True, "plausible": False, "proposed_plan": []}
                
                plan["proposed_plan"] = valid_plan
                return plan
            return None
        except Exception as e:
            print(f"  -> LLM_ERROR: {e}") 
            return None