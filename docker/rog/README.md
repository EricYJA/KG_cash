# RoG inference container

Serves [rmanluo/RoG](https://huggingface.co/rmanluo/RoG) (fine-tuned LLaMA-2-7B) as an HTTP API.

## Prereqs

- NVIDIA GPU with driver >=520 and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  installed on the host (`nvidia-smi` works from inside a plain CUDA container).
- HuggingFace token with LLaMA-2 access accepted (see the walkthrough in the main setup notes).

## Setup

Export your token in the shell that runs `docker compose`:

```bash
export HF_TOKEN=hf_...
```

## Bring it up

From the repo root:

```bash
docker compose up -d --build rog
docker compose logs -f rog
```

First run downloads the ~13 GB model to `~/.cache/huggingface` on the host (mounted into the container). Subsequent runs are instant.

Startup takes ~1-3 minutes on a warm cache while the model loads into GPU memory in 8-bit.

## Test

```bash
curl -s http://localhost:8080/health | python -m json.tool

curl -s -X POST http://localhost:8080/plan \
    -H 'content-type: application/json' \
    -d '{"question": "what is the name of justin bieber brother", "n_beam": 3}' \
    | python -m json.tool
```

Expected `/plan` response shape:

```json
{
  "paths": [
    ["people.person.parents", "people.person.children"],
    ["people.person.sibling_s", "people.sibling_relationship.sibling"]
  ],
  "raw_outputs": ["...<PATH> people.person.parents <SEP> ... </PATH>", ...],
  "scores": [-0.42, -0.51, -0.68],
  "norm_scores": [0.42, 0.31, 0.27]
}
```

## Endpoints

- `GET /health` — liveness + model/quant info.
- `POST /generate` — generic beam-search generation. Body: `{prompt, max_new_tokens, num_beams, do_sample, temperature, top_p, return_scores}`. Use for reasoner step with your own prompt.
- `POST /plan` — RoG planner shortcut. Body: `{question, n_beam, max_new_tokens}`. Wraps the question in RoG's INSTRUCTION prompt and parses `<PATH>...<SEP>...</PATH>` output into relation lists.

## Config knobs (env vars in docker-compose.yml)

- `MODEL_ID` — swap the checkpoint (default `rmanluo/RoG`).
- `QUANTIZATION` — `8bit` (default), `4bit`, or `fp16`.
- `PORT` — internal port (default 8080).

## Common issues

- **`no kernel image is available for execution on the device`** — bitsandbytes kernel doesn't support your GPU arch. On Pascal (GTX 10xx), `bitsandbytes==0.42.0` is already pinned; if you upgraded, revert.
- **`401 Repository not found`** on model download — HF_TOKEN missing or LLaMA-2 license not accepted.
- **`CUDA out of memory` on startup** — switch `QUANTIZATION` to `4bit` in `docker-compose.yml`.
