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


# Interchangeable SPARQL backends serving the same Freebase KG. Select with
# KG_BACKEND=<name>; the chosen endpoint is exported as SPARQL_ENDPOINT so the
# ToG subprocesses (freebase_func.py) pick it up. SPARQL_ENDPOINT set
# explicitly in the environment always wins.
KG_BACKENDS = {
    "virtuoso": {"service": "virtuoso", "endpoint": "http://localhost:8890/sparql"},
    "oxigraph": {"service": "oxigraph", "endpoint": "http://localhost:7878/query"},
}


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


def samples_flag(limit) -> str:
    """Translate a --limit value into main_tog2.py's --samples argument.

    'all' (case-insensitive) means the whole dataset; main_tog2.py takes an int
    and clamps with min(samples, len(datas)), so a value larger than any dataset
    runs every question. Anything else is passed through unchanged.
    """
    if str(limit).strip().lower() == "all":
        return str(10**9)
    return str(limit)


def add_run_args(p) -> None:
    """Flags every ToG runner needs to coexist with a second, concurrent instance.

    --kg-backend picks which SPARQL server to query; --run-tag namespaces the
    default output paths so two instances never write to the same file. The tag
    defaults to the backend name, which is the usual reason to run two at once.
    """
    p.add_argument("--kg-backend", default=os.environ.get("KG_BACKEND", "virtuoso"),
                   choices=sorted(KG_BACKENDS),
                   help="SPARQL backend to query (default: virtuoso)")
    p.add_argument("--run-tag", default=os.environ.get("RUN_TAG", ""),
                   help="suffix for default output paths (default: the --kg-backend name)")


def resolve_run(args) -> tuple[str, str]:
    """Bring up the selected backend; return (endpoint, output tag)."""
    endpoint = ensure_kg_backend(args.kg_backend, explicit=True)
    return endpoint, (args.run_tag or args.kg_backend)


def ensure_kg_backend(name: str | None = None, *, explicit: bool = False) -> str:
    """Bring up a SPARQL backend service and wait until it answers.

    Picks the backend from `name`, else the KG_BACKEND env var, else virtuoso.
    Exports the endpoint as SPARQL_ENDPOINT and returns it.

    With `explicit` (the --kg-backend path) the chosen backend overwrites any
    inherited SPARQL_ENDPOINT: otherwise a stray value in .env would quietly aim
    both concurrent instances at the same server, and the comparison would be a
    lie rather than an error.
    """
    backend = name or os.environ.get("KG_BACKEND", "virtuoso")
    if backend not in KG_BACKENDS:
        sys.exit(f"unknown KG_BACKEND {backend!r}; choose from {sorted(KG_BACKENDS)}")
    service = KG_BACKENDS[backend]["service"]
    default_endpoint = KG_BACKENDS[backend]["endpoint"]
    if explicit:
        endpoint = os.environ["SPARQL_ENDPOINT"] = default_endpoint
    else:
        endpoint = os.environ.setdefault("SPARQL_ENDPOINT", default_endpoint)
    ping = endpoint + "?query=ASK%20%7B%7D"

    print(f">>> starting {backend} (Freebase SPARQL endpoint at {endpoint})", flush=True)
    subprocess.run(["docker", "compose", "up", "-d", service], cwd=REPO_ROOT, check=True)
    print(f">>> waiting for {backend} to answer SPARQL ...", flush=True)
    for _ in range(90):
        try:
            with urllib.request.urlopen(ping, timeout=3) as resp:
                if resp.status == 200:
                    print(f">>> {backend} is up", flush=True)
                    return endpoint
        except Exception:
            pass
        time.sleep(2)
    sys.exit(f"{backend} did not become ready at {endpoint}")


def ensure_virtuoso() -> None:
    """Backward-compatible alias: bring up the backend selected by KG_BACKEND."""
    ensure_kg_backend()
