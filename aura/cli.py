# aura/cli.py
import argparse
from pathlib import Path
import json


def detect_installable(repo_root: Path) -> bool:
    """Return True if setup.py or setup.cfg exists in this repo."""
    has_setup_py = (repo_root / "setup.py").exists()
    has_setup_cfg = (repo_root / "setup.cfg").exists()
    return has_setup_py or has_setup_cfg


def ensure_requirements(repo_root: Path) -> str:
    """Ensure requirements.txt exists, or exit."""
    req = repo_root / "requirements.txt"
    if not req.exists():
        print(
            "ERROR: requirements.txt not found in this repository.\n"
            "AURA requires a requirements.txt file to install dependencies in CI.\n"
            "Please create one and rerun `aura init`."
        )
        raise SystemExit(1)
    return "requirements.txt"


def create_default_validation_script(
    repo_root: Path,
    script_path: str,
    project_name: str,
    is_installable: bool,
) -> None:
    """Create a minimal validation_aura.py smoke test."""
    full_path = repo_root / script_path

    if is_installable:
        # Try to import the project package; user can extend further.
        body = f"""\"\"\"Minimal smoke test created by AURA.

This script is a starting point for validating your project after
dependency upgrades. AURA runs it twice in CI:

- once in the baseline environment (current requirements.txt)
- once in the upgraded environment (primary_requirements.txt)

If the upgraded run fails but the baseline run passes, AURA will NOT
submit the upgrade PR.

Right now this script only checks that your main package can be
imported. You should extend it with real checks that reflect how you
actually use this project in production or CI.
\"\"\"

def main():
    print("Running AURA validation smoke test for '{project_name}'...")
    try:
        __import__("{project_name}")
        print("Import of '{project_name}' succeeded.")
    except Exception as exc:
        raise SystemExit(f"Import of '{project_name}' failed: {{exc}}")

    # TODO: Add additional imports / function calls here that exercise
    #       the critical behavior of your project.


if __name__ == "__main__":
    main()
"""
    else:
        # Non-installable: generic placeholder.
        body = """\"\"\"Minimal smoke test created by AURA.

This script is a starting point for validating your project after
dependency upgrades. AURA runs it twice in CI:

- once in the baseline environment (current requirements.txt)
- once in the upgraded environment (primary_requirements.txt)

If the upgraded run fails but the baseline run passes, AURA will NOT
submit the upgrade PR.

Right now this script only prints a success message. You should replace
this with imports and function calls that match how you actually use
this repository.
\"\"\"

def main():
    print("AURA validation smoke test ran successfully.")
    # TODO: Add imports and calls that exercise your project here.


if __name__ == "__main__":
    main()
"""

    full_path.write_text(body, encoding="utf-8")
    print(f"Created {script_path}.\n")
    print(
        "AURA created a minimal validation script at "
        f"'{script_path}'.\n"
        "Please edit this file to exercise your project's main behavior "
        "before relying on AURA's upgrade decisions.\n"
    )


def ask_validation_config(repo_root: Path, is_installable: bool) -> dict:
    """Interactively collect validation configuration (script or pytest)."""
    print("Step 3: Configure validation / tests for AURA.\n")

    has_tests = input(
        "Do you already have a validation or test entry point? (y/n): "
    ).strip().lower()

    if has_tests == "y":
        kind = input(
            "Is it (1) a normal Python script (run with `python your_script.py`) "
            "or (2) a pytest suite? [1/2]: "
        ).strip()

        if kind == "1":
            script_path = input(
                "Enter the script path relative to the repo root "
                "(e.g., tests/validation.py): "
            ).strip()
            return {
                "type": "script",
                "smoke_test_script": script_path,
                "project_dir": ".",
            }

        # treat anything else as pytest for now
        pytest_target = input(
            "Enter the pytest target (file or folder, e.g., tests/ or tests/test_integration.py): "
        ).strip()
        return {
            "type": "pytest",
            "pytest_target": pytest_target,
            "project_dir": ".",
        }

    # No tests -> offer to create a minimal script
    print(
        "You do not have a validation/test entry point.\n"
        "AURA can create a minimal smoke-test script for you."
    )
    create = (
        input(
            "Do you want AURA to create validation_aura.py now? (y/n) [y]: "
        )
        .strip()
        .lower()
        or "y"
    )

    if create != "y":
        print(
            "Aborting. Please add a validation script or pytest suite "
            "and rerun `aura init`."
        )
        raise SystemExit(1)

    script_path = "validation_aura.py"
    project_name = repo_root.name.replace("-", "_")
    create_default_validation_script(
        repo_root=repo_root,
        script_path=script_path,
        project_name=project_name,
        is_installable=is_installable,
    )

    return {
        "type": "script",
        "smoke_test_script": script_path,
        "project_dir": ".",
    }


def ask_schedule_preset() -> str | None:
    """Ask how often to run AURA in GitHub Actions."""
    print("\nStep 4: Configure how AURA runs in GitHub Actions.\n")
    print("How do you want AURA to run in GitHub Actions?")
    print("  1) Manually (only when you click 'Run workflow')")
    print("  2) On a schedule (plus manual runs)")
    choice = (input("Select an option [1/2, default: 1]: ").strip() or "1")

    if choice != "2":
        return None  # manual only

    print("\nHow often should AURA run?")
    print("  1) Daily")
    print("  2) Weekly")
    freq_choice = (input("Select an option [1/2, default: 1]: ").strip() or "1")
    return "weekly" if freq_choice == "2" else "daily"


def build_agent_config(
    project_name: str,
    is_installable: bool,
    requirements_file: str,
    validation_config: dict,
) -> str:
    """Build AGENT_CONFIG as a Python dict literal string."""
    config = {
        "PROJECT_NAME": project_name,
        "IS_INSTALLABLE_PACKAGE": is_installable,
        "REQUIREMENTS_FILE": requirements_file,
        "METRICS_OUTPUT_FILE": "metrics_output.txt",
        "PRIMARY_REQUIREMENTS_FILE": "primary_requirements.txt",
        "VALIDATION_CONFIG": validation_config,
        "MAX_RUN_PASSES": 3,
    }
    cfg = json.dumps(config, indent=4)
    # convert JSON booleans/null to Python style
    cfg = cfg.replace("true", "True").replace("false", "False").replace("null", "None")
    return cfg


def write_dependency_agent(
    repo_root: Path,
    is_installable: bool,
    requirements_file: str,
    validation_config: dict,
):
    """Create dependency_agent.py in the target repo with a filled AGENT_CONFIG."""
    project_name = repo_root.name

    agent_config_str = build_agent_config(
        project_name=project_name,
        is_installable=is_installable,
        requirements_file=requirements_file,
        validation_config=validation_config,
    )

    body = f'''import os
import sys
from google import genai

from aura.agent_logic import DependencyAgent
from aura.dependency_agent_template import GeminiClientWrapper


AGENT_CONFIG = {agent_config_str}


if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        sys.exit("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")

    llm_client = GeminiClientWrapper(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.5-flash",
    )

    agent = DependencyAgent(config=AGENT_CONFIG, llm_client=llm_client)
    agent.run()
'''
    target_path = repo_root / "dependency_agent.py"
    target_path.write_text(body, encoding="utf-8")
    print(f"Wrote dependency_agent.py to {target_path}")


def write_github_workflow(
    repo_root: Path,
    validation_config: dict,
    schedule_preset: str | None,
):
    print("\nStep 5: Configure GitHub Actions workflow for AURA.\n")
    python_version = input(
        "Which Python version should AURA use in CI? (e.g., 3.10, 3.11) [default: 3.11]: "
    ).strip() or "3.11"

    vtype = validation_config.get("type", "script")
    if vtype == "script":
        smoke_script = validation_config.get("smoke_test_script", "validation.py")
        baseline_cmd = f"./baseline_env/bin/python {smoke_script}"
        upgrade_cmd = f"./upgrade_env/bin/python {smoke_script}"
    else:  # pytest
        pytest_target = validation_config.get("pytest_target", "tests")
        baseline_cmd = f"./baseline_env/bin/python -m pytest {pytest_target}"
        upgrade_cmd = f"./upgrade_env/bin/python -m pytest {pytest_target}"

    # on: block with manual + optional schedule
    on_lines = [
        "on:",
        "  workflow_dispatch:",
    ]
    if schedule_preset == "daily":
        on_lines.append("  schedule:")
        on_lines.append("    - cron: '0 3 * * *'  # daily at 03:00 UTC")
    elif schedule_preset == "weekly":
        on_lines.append("  schedule:")
        on_lines.append("    - cron: '0 3 * * 0'  # weekly on Sunday 03:00 UTC")
    on_block = "\n".join(on_lines)

    workflow_dir = repo_root / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = workflow_dir / "aura.yml"

    yml = f"""name: AURA Speed Run

{on_block}

jobs:
  aura-modernize:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '{python_version}'

      - name: Install AURA agent
        run: |
          python -m venv aura_env
          ./aura_env/bin/pip install --upgrade pip
          ./aura_env/bin/pip install aura-dep-agent


      - name: 1. Run AURA Agent
        run: |
          python -m venv --system-site-packages aura_env
          ./aura_env/bin/pip install --upgrade pip
          ./aura_env/bin/pip install aura-dep-agent
          ./aura_env/bin/python -u dependency_agent.py 
        env:
          GEMINI_API_KEY: ${{{{ secrets.GEMINI_API_KEY }}}}

      - name: Final Commit
        if: always()
        run: |
          git config --global user.name 'AURA Agent'
          git config --global user.email 'aura-agent@users.noreply.github.com'
          if ! git diff --staged --quiet; then
            git commit -m ": Modernized dependencies [skip ci]"
            git push
          fi
"""
    workflow_path.write_text(yml, encoding="utf-8")
    print(f"Wrote GitHub Actions workflow to {workflow_path}")


def aura_init():
    repo_root = Path(".").resolve()
    print(f"\nAURA init running in {repo_root}\n")

    # Step 1: installable vs non-installable
    is_installable = detect_installable(repo_root)
    if is_installable:
        print(
            "Detected setup.py/setup.cfg -> treating project as an installable package (IS_INSTALLABLE_PACKAGE=True)."
        )
    else:
        print(
            "No setup.py or setup.cfg found -> treating project as non-installable (IS_INSTALLABLE_PACKAGE=False)."
        )

    # Step 2: requirements.txt
    requirements_file = ensure_requirements(repo_root)
    print(f"Found {requirements_file}.\n")

    # Step 3: validation config
    validation_config = ask_validation_config(repo_root, is_installable)
    print(f"Validation configuration selected: {validation_config}\n")

    # Step 4: schedule preset
    schedule_preset = ask_schedule_preset()

    # Step 5: write dependency_agent.py into this repo
    write_dependency_agent(
        repo_root=repo_root,
        is_installable=is_installable,
        requirements_file=requirements_file,
        validation_config=validation_config,
    )

    # Step 6: write GitHub Actions workflow
    write_github_workflow(repo_root, validation_config, schedule_preset)

    print(
        "\nAURA init completed successfully.\n"
        "Next steps:\n"
        "  1) Commit and push dependency_agent.py, validation_aura.py "
        "     (if created), and .github/workflows/aura.yml\n"
        "  2) In GitHub, go to Actions → 'AURA Speed Run' → 'Run workflow' "
        "to start your first run.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize AURA in this repo")

    args = parser.parse_args()
    if args.command == "init":
        aura_init()
    else:
        parser.print_help()
