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
KEYS_PATH = REPO_ROOT / ".env_keys"
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


def load_env_keys() -> int:
    """Point the LLM client at repo-root .env_keys; return how many keys it holds.

    The client falls back to the next key on a 400/5xx answer, so handing it the
    whole pool is what keeps a long run alive when one key is exhausted. 0 means
    there is no key file and the run falls back to the single LLM_API_KEY.
    """
    if not KEYS_PATH.exists():
        return 0
    os.environ.setdefault("LLM_KEYS_FILE", str(KEYS_PATH))
    return sum(
        1
        for line in KEYS_PATH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    )


# Which env file the runners read. A repo like this one accumulates a file per
# configuration (.env_gemini, .env_oxi_gpt, ...); ENV_FILE or --env-file picks
# one. Sourcing such a file in the shell does NOT work: without `export` the
# values stay shell-local and never reach the Python subprocess that does the
# work, which is how a run labelled for one model silently used another.
ENV_FILE_ENV = "ENV_FILE"

# The env file actually loaded, so the later validating load_dotenv() checks the
# same file preload_dotenv() read rather than falling back to .env.
_LOADED_ENV_FILE: Path | None = None


def _parse_env_line(line: str):
    """(key, value) from one env-file line, or None for a blank/comment line.

    Handles the two things these files actually contain beyond bare KEY=VALUE:
    quoted values (MODEL="protected.gemini-3.1-flash-lite") and trailing inline
    comments (KG_BACKEND=oxigraph  # virtuoso | oxigraph). A '#' only starts a
    comment when whitespace precedes it, so a '#' inside a value -- a URL
    fragment, a key -- survives. `export KEY=VALUE` is accepted too, since a
    file written to be sourced may well have it.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key, value = key.strip(), value.strip()
    if not key:
        return None
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        value = value[1:end] if end != -1 else value[1:]
    else:
        for i, ch in enumerate(value):
            if ch == "#" and i and value[i - 1] in " \t":
                value = value[:i]
                break
        value = value.strip()
    return key, value


def resolve_env_file(explicit: str | None = None) -> Path:
    """Which env file to load: explicit > --env-file > ENV_FILE > repo-root .env.

    A bare name or a path relative to the caller's cwd both work, and a name
    that is not found there is looked for in the repo root -- so
    `--env-file .env_gemini` behaves the same from the repo root, from scripts/,
    or from anywhere else.
    """
    candidate = (explicit or env_file_from_argv()
                 or os.environ.get(ENV_FILE_ENV) or ".env")
    direct = Path(candidate)
    if direct.is_file():
        return direct
    in_repo = REPO_ROOT / candidate
    return in_repo if in_repo.is_file() else direct


def env_file_from_argv(argv: list[str] | None = None) -> str | None:
    """--env-file read straight out of argv, before any parser exists.

    argparse evaluates a flag's default when the flag is declared, so the env
    file has to be in os.environ before the parser is built -- which is before
    argparse could report which file was requested. The flag is still declared
    on the parser (add_env_file_arg) so it appears in --help and is not rejected
    as unknown.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, arg in enumerate(argv):
        if arg == "--env-file" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--env-file="):
            return arg.split("=", 1)[1]
    return None


def add_env_file_arg(p) -> None:
    """Declare --env-file so it shows in --help and is not an unknown arg."""
    p.add_argument("--env-file", default=os.environ.get(ENV_FILE_ENV, ""),
                   help="env file holding this run's settings (MODEL, VENDOR, "
                        "KG_BACKEND, RUN_TAG, ...). Default: .env in the repo "
                        "root. Read directly by the runner -- do not `source` "
                        "it, that does not reach the subprocess.")


def preload_dotenv(path: str | None = None) -> Path:
    """Put the chosen env file into os.environ, validating nothing.

    The runners' flags take their defaults from the environment
    (`default=env("MODEL", "")`, `env("KG_BACKEND", ...)`, ...) and argparse
    evaluates a default when the argument is declared, so this has to run before
    the parser is built or those values are ignored. Real environment variables
    win over the file (setdefault), so an exported override still takes effect.

    Deliberately silent about a missing file and about empty keys: `--help` has
    to keep working, and the validating load_dotenv(required=...) call later in
    main() is what refuses to start a run with no key. Returns the file it read.
    """
    global _LOADED_ENV_FILE
    requested = path or env_file_from_argv() or os.environ.get(ENV_FILE_ENV)
    env_path = resolve_env_file(path)
    _LOADED_ENV_FILE = env_path
    if not env_path.is_file():
        if requested:
            # Asking for a specific file and getting the defaults instead is the
            # failure this whole mechanism exists to stop: it is how a run gets
            # labelled for one model and executed with another. Only the
            # fallback to a missing .env stays quiet, so --help still works.
            available = sorted(p.name for p in REPO_ROOT.glob(".env*")
                               if p.is_file() and p.name != ".env.example")
            sys.exit(
                f"env file {requested!r} not found (looked at {Path(requested)} "
                f"and {REPO_ROOT / requested}).\n"
                f"In the repo root: {', '.join(available) or '(none)'}\n"
                f"Pass a bare name, e.g. --env-file .env_gemini -- it is "
                f"resolved against the repo root from any directory."
            )
        return env_path
    for line in env_path.read_text().splitlines():
        parsed = _parse_env_line(line)
        if parsed:
            os.environ.setdefault(*parsed)
    return env_path


def load_dotenv(required: tuple[str, ...] = (), path: str | None = None) -> None:
    """Load the chosen env file into os.environ (no override) and validate.

    Exits if the file is missing or any `required` key is empty/unset. Defaults
    to whatever preload_dotenv() already read, so both calls in a run always
    agree on which file this run is configured by.
    """
    env_path = resolve_env_file(path) if path else (_LOADED_ENV_FILE
                                                   or resolve_env_file())
    if not env_path.is_file():
        sys.exit(f"missing {env_path} (need: {', '.join(required) or '<none>'})")
    preload_dotenv(str(env_path))
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
