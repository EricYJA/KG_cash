"""Shared plumbing for the RoG cache experiment runners (scripts/run_rog_*.py).

RoG runs fully containerized on the GPU (the kgcash/rog-eval image pins the same
torch/bitsandbytes as the server), so these helpers wrap `docker build` and
`docker run` rather than a local interpreter. Standard library only.

--network host: this box has net.ipv4.ip_forward=0, so bridged containers cannot
reach huggingface.co (see docker-compose.yml).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KEYS_PATH = REPO_ROOT / ".env_keys"
IMAGE = "kgcash/rog-eval:latest"
DOCKERFILE = "docker/rog/Dockerfile.eval"
BUILD_CONTEXT = "docker/rog"

MODEL_NAME = "RoG"
MODEL_PATH = "rmanluo/RoG"

# Where RoG's KG lookups go. "dataset" is upstream's behaviour: every KG read
# comes from the per-question subgraph bundled in the HuggingFace row, which was
# built by someone who already knew the answer. The other two are the live
# Freebase endpoints ToG has always queried (see scripts/_tog_common.py).
#
# Worth stating plainly because the artifacts do not: every RoG run under
# artifacts/rog_cache/ predating this flag used "dataset", including the ones
# tagged *_virtuoso_* and *_oxi_*. Those tags recorded which endpoint the
# concurrent ToG run was using; RoG itself had no SPARQL path at all.
KG_BACKENDS = ("dataset", "virtuoso", "oxigraph")


def assert_stages_agree(stage1_stats: Path, stage2_config: Path, requested: str) -> None:
    """Fail unless the planner and the reasoner actually ran the same model.

    Both stages resolve their own LLM config, so "the model" is two independent
    resolutions of one --model flag, and an omitted flag resolves to the vendor
    preset in silence. That silence has already cost this project real runs:
    artifacts/rog_cache/gemini_rog_cache_virtuoso_2 and _3 are tagged gemini and
    were planned with Claude-Haiku, and both parametric "gemini" runs recorded
    `model: null`. A run tag is not evidence; these two files are.

    Reads what each stage wrote (stage 1: cache_stats.json, stage 2:
    run_config.json) rather than what the runner asked for, so it catches a flag
    that never arrived as well as one that arrived differently.
    """
    def _read(path, keys):
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return tuple(data.get(k) for k in keys)

    stage1 = _read(stage1_stats, ("vendor", "model"))
    stage2 = _read(stage2_config, ("vendor", "model"))
    if stage1 is None or stage2 is None:
        missing = stage1_stats if stage1 is None else stage2_config
        print(f"    [warn] cannot verify the model across stages: {missing} is "
              f"missing or unreadable")
        return

    if stage1 != stage2:
        sys.exit(
            f"planner and reasoner ran DIFFERENT models -- this run is not "
            f"attributable to either.\n"
            f"  stage 1 (planner):  {stage1[0]}/{stage1[1]}\n"
            f"  stage 2 (reasoner): {stage2[0]}/{stage2[1]}"
        )
    if requested and stage1[1] != requested:
        sys.exit(
            f"--model {requested!r} was requested but both stages resolved "
            f"{stage1[1]!r}. Check the vendor supports that model id."
        )
    print(f"    both stages ran {stage1[0]}/{stage1[1]}"
          + ("" if requested else "  (vendor preset default; pass --model to pin one)"))


def add_kg_backend_arg(parser) -> None:
    """Add --kg-backend to a RoG runner, defaulting to the live Virtuoso."""
    parser.add_argument(
        "--kg-backend", default=os.environ.get("KG_BACKEND", "virtuoso"),
        choices=KG_BACKENDS,
        help="where stage 1 and stage 2 read the KG. virtuoso (default) and "
             "oxigraph query the live Freebase endpoint; dataset uses the "
             "per-question subgraph bundled in the HuggingFace row, which is "
             "upstream RoG's behaviour and what every earlier run here used.",
    )


def ensure_kg_backend(backend: str) -> str | None:
    """Start the SPARQL service for `backend` and wait for it; return its endpoint.

    None for "dataset", which needs no server. Delegates to the ToG helper so both
    halves of the project bring up the same containers the same way and cannot
    disagree about which URL "virtuoso" means.
    """
    if backend == "dataset":
        print(">>> KG backend: dataset (per-question bundled subgraph; no endpoint)",
              flush=True)
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _tog_common import ensure_kg_backend as tog_ensure_kg_backend

    return tog_ensure_kg_backend(backend, explicit=True)


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


def split_for(n: str) -> str:
    """'all' -> full test split; otherwise the first n rows."""
    return "test" if n == "all" else f"test[:{n}]"


def quant_flag(quant: str) -> list[str]:
    """Map a quant name to RoG's CLI flag. fp16 will NOT fit in 11GB."""
    mapping = {"8bit": ["--load_in_8bit"], "4bit": ["--load_in_4bit"], "fp16": []}
    if quant not in mapping:
        sys.exit("QUANT/--quant must be 8bit|4bit|fp16")
    return mapping[quant]


def docker_build(quiet: bool = False) -> None:
    print(">>> building eval image", flush=True)
    cmd = ["docker", "build", "--network=host", "-f", DOCKERFILE, "-t", IMAGE, BUILD_CONTEXT]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                   stdout=subprocess.DEVNULL if quiet else None)


def stop_rog_server() -> None:
    """The server holds ~7GB of VRAM; free the whole card before a GPU run."""
    out = subprocess.run(["docker", "ps", "-q", "-f", "name=kgcash-rog"],
                         cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if out:
        print(">>> stopping kgcash-rog to free the GPU "
              "(restart later with: docker compose up -d rog)", flush=True)
        subprocess.run(["docker", "compose", "stop", "rog"], cwd=REPO_ROOT, check=True)


def make_rog_runner(*, mounts: list[str], pythonpath: str, gpus: bool, use_user: bool,
                    extra_env: dict[str, str] | None = None, workdir: str = "/rog"):
    """Build a `rog(args)` closure that runs `docker run ... IMAGE <args>`.

    `--user` (experiment/sim) keeps artifacts host-owned so the host user can
    delete them; it also forces HOME/HF_HOME off /root, which is not writable as a
    non-root user. `mounts` are raw `src:dst[:ro]` strings.
    """
    hf_token = os.environ["HF_TOKEN"]

    def rog(args: list[str]) -> None:
        cmd = ["docker", "run", "--rm", "-i"]
        if gpus:
            cmd += ["--gpus", "all"]
        cmd += ["--network", "host"]
        if use_user:
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}",
                    "-e", "HOME=/tmp", "-e", "HF_HOME=/hf"]
        cmd += ["-e", f"HF_TOKEN={hf_token}", "-e", f"HUGGING_FACE_HUB_TOKEN={hf_token}",
                "-e", f"PYTHONPATH={pythonpath}"]
        for key, value in (extra_env or {}).items():
            cmd += ["-e", f"{key}={value}"]
        for mount in mounts:
            cmd += ["-v", mount]
        cmd += ["-w", workdir, IMAGE, *[str(a) for a in args]]
        print(f"\n[docker] {' '.join(cmd[cmd.index(IMAGE) + 1:])}", flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    return rog
