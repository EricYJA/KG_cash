import os
import re
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = os.environ.get("MODEL_ID", "rmanluo/RoG")
QUANTIZATION = os.environ.get("QUANTIZATION", "8bit").lower()
HF_TOKEN = os.environ.get("HF_TOKEN") or True

PATH_RE = re.compile(r"<PATH>(.*?)</PATH>")

STATE: dict = {}


def _quant_config():
    if QUANTIZATION == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if QUANTIZATION == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, token=HF_TOKEN)
    kwargs = {"device_map": "auto", "token": HF_TOKEN}
    quant = _quant_config()
    if quant is not None:
        kwargs["quantization_config"] = quant
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    STATE["tokenizer"] = tokenizer
    STATE["model"] = model
    yield
    STATE.clear()


app = FastAPI(title="RoG inference", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    num_beams: int = 1
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    return_scores: bool = False


class GenerateResponse(BaseModel):
    outputs: list[str]
    scores: Optional[list[float]] = None
    norm_scores: Optional[list[float]] = None


class PlanRequest(BaseModel):
    question: str
    n_beam: int = Field(3, ge=1, le=8)
    max_new_tokens: int = 100


class PlanResponse(BaseModel):
    paths: list[list[str]]
    raw_outputs: list[str]
    scores: list[float]
    norm_scores: list[float]


@app.get("/health")
def health():
    return {
        "model": MODEL_ID,
        "quantization": QUANTIZATION,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "loaded": "model" in STATE,
    }


@app.post("/generate", response_model=GenerateResponse)
@torch.inference_mode()
def generate(req: GenerateRequest):
    model = STATE.get("model")
    tokenizer = STATE.get("tokenizer")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    input_ids = tokenizer.encode(req.prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=req.max_new_tokens,
        num_beams=req.num_beams,
        num_return_sequences=req.num_beams,
        do_sample=req.do_sample,
        return_dict_in_generate=True,
        output_scores=req.return_scores or req.num_beams > 1,
        early_stopping=False,
    )
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p

    output = model.generate(**gen_kwargs)
    decoded = tokenizer.batch_decode(
        output.sequences[:, input_ids.shape[1]:], skip_special_tokens=True
    )
    decoded = [d.strip() for d in decoded]

    resp = GenerateResponse(outputs=decoded)
    if req.num_beams > 1 and getattr(output, "sequences_scores", None) is not None:
        scores = output.sequences_scores
        resp.scores = scores.tolist()
        resp.norm_scores = torch.softmax(scores, dim=0).tolist()
    return resp


INSTRUCTION = (
    "Please generate a valid relation path that can be helpful for answering "
    "the following question: "
)

# RoG is a LLaMA-2-chat fine-tune; it only emits <PATH>...</PATH> when the
# instruction is wrapped in the chat template it was trained with. Verbatim from
# RoG's prompts/llama2.txt (applied by InstructFormater in gen_rule_path.py).
PROMPT_TEMPLATE = "[INST] <<SYS>>\n<</SYS>>\n{instruction}{input} [/INST]"


def _parse_paths(raw_outputs: list[str]) -> list[list[str]]:
    result = []
    for text in raw_outputs:
        match = PATH_RE.search(text)
        if match is None:
            continue
        parts = [p.strip() for p in match.group(1).split("<SEP>") if p.strip()]
        if parts:
            result.append(parts)
    return result


@app.post("/plan", response_model=PlanResponse)
@torch.inference_mode()
def plan(req: PlanRequest):
    """Convenience endpoint that mirrors RoG's `gen_rule_path.py` planner:
    wraps the question in the RoG INSTRUCTION prompt, runs beam search,
    and returns parsed relation paths + scores."""
    prompt = PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=req.question)
    gen = generate(
        GenerateRequest(
            prompt=prompt,
            max_new_tokens=req.max_new_tokens,
            num_beams=req.n_beam,
            do_sample=False,
            return_scores=True,
        )
    )
    return PlanResponse(
        paths=_parse_paths(gen.outputs),
        raw_outputs=gen.outputs,
        scores=gen.scores or [1.0] * len(gen.outputs),
        norm_scores=gen.norm_scores or [1.0 / max(len(gen.outputs), 1)] * len(gen.outputs),
    )
