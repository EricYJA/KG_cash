"""Shared plumbing for the RoG cache experiment runners (scripts/run_rog_*.py).

RoG runs fully containerized on the GPU (the kgcash/rog-eval image pins the same
torch/bitsandbytes as the server), so these helpers wrap `docker build` and
`docker run` rather than a local interpreter. Standard library only.

--network host: this box has net.ipv4.ip_forward=0, so bridged containers cannot
reach huggingface.co (see docker-compose.yml).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE = "kgcash/rog-eval:latest"
DOCKERFILE = "docker/rog/Dockerfile.eval"
BUILD_CONTEXT = "docker/rog"

MODEL_NAME = "RoG"
MODEL_PATH = "rmanluo/RoG"


def load_dotenv(required: tuple[str, ...] = ()) -> None:
    """Load KEY=VALUE lines from repo-root .env into os.environ (no override)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        sys.exit(f"missing {env_path} (need: {', '.join(required) or '<none>'})")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
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
