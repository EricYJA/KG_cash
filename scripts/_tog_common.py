"""Shared plumbing for the ToG cache experiment runners (scripts/run_tog_*.py).

These orchestrator scripts can run under any Python; the actual ToG code runs in
the `KG_cash` conda env (which has sentence_transformers, SPARQLWrapper, ...),
invoked as a subprocess. This module only uses the standard library.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOG_DIR = REPO_ROOT / "src" / "ToG-cache" / "ToG"
EVAL_DIR = REPO_ROOT / "src" / "ToG-cache" / "eval"
SPARQL_PING = "http://localhost:8890/sparql?query=ASK%20%7B%7D"


def python_cmd(conda_env: str | None = None) -> list[str]:
    """Return the argv prefix that runs Python inside the target conda env.

    Prefers the env's interpreter directly (`.../envs/<env>/bin/python`); falls
    back to `conda run` if only the `conda` launcher can be found.
    """
    env = conda_env or os.environ.get("CONDA_ENV", "KG_cash")

    bases: list[Path] = []
    for var in ("CONDA_ROOT", "CONDA_PREFIX_1", "MAMBA_ROOT_PREFIX"):
        val = os.environ.get(var)
        if val:
            bases.append(Path(val))
    bases += [Path.home() / "anaconda3", Path.home() / "miniconda3", Path("/opt/conda")]
    if shutil.which("conda"):
        try:
            base = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
            bases.append(Path(base))
        except Exception:
            pass

    for base in bases:
        candidate = base / "envs" / env / "bin" / "python"
        if candidate.exists():
            return [str(candidate)]

    if shutil.which("conda"):
        return ["conda", "run", "--no-capture-output", "-n", env, "python"]

    sys.exit(
        f"could not find a Python for conda env {env!r}; "
        f"install it, activate it, or set CONDA_ENV to an existing env"
    )


def run_py(args: list[str], cwd: Path, conda_env: str | None = None) -> None:
    """Run `python <args>` in the target conda env, streaming output live."""
    cmd = python_cmd(conda_env) + [str(a) for a in args]
    print(f"\n[run] (cwd={cwd}) {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_dotenv(required: tuple[str, ...] = ()) -> None:
    """Load KEY=VALUE lines from repo-root .env into os.environ (no override).

    Exits if .env is missing or any `required` key is empty/unset.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        sys.exit(f"missing {env_path} (need: {', '.join(required) or '<none>'})")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
    for key in required:
        if not os.environ.get(key):
            sys.exit(f"{key} not set in {env_path}")


def ensure_virtuoso() -> None:
    """Bring up the Virtuoso (Freebase SPARQL) service and wait until it answers."""
    print(">>> starting Virtuoso (Freebase SPARQL endpoint on :8890)", flush=True)
    subprocess.run(["docker", "compose", "up", "-d", "virtuoso"], cwd=REPO_ROOT, check=True)
    print(">>> waiting for Virtuoso to answer SPARQL ...", flush=True)
    for _ in range(90):
        try:
            with urllib.request.urlopen(SPARQL_PING, timeout=3) as resp:
                if resp.status == 200:
                    print(">>> Virtuoso is up", flush=True)
                    return
        except Exception:
            pass
        time.sleep(2)
    sys.exit("Virtuoso did not become ready on :8890")
