"""RoG stage 2 (reasoner) against a chat LLM API, with a seeded RNG.

    relation paths + KG  ->  answer          (never cached)

Deliberately thin: it wraps the LLM API in RoG's own BaseLanguageModel interface
and then calls RoG's `predict_answer.main()` unchanged. Prompt construction,
path grounding, output layout and scoring therefore stay byte-identical to
upstream -- the only difference from `python src/qa_prediction/predict_answer.py`
is which model answers, which is exactly what we want when attributing an
accuracy delta to the *planner* cache.

Why --seed: RoG's PromptBuilder.check_prompt_length shuffles and truncates
overflowing path lists with the unseeded global `random` (build_qa_input.py:126).
That alone swings Hits@1 by ~10 points run-to-run and would swamp the caching
effect being measured, so every run pins it.

Usage (inside the rog-eval image; see scripts/run_rog_cache_experiment.py):
    python predict_answer_api.py --model_name RoG -d RoG-webqsp --split test \
        --prompt_path prompts/llama2_predict.txt --add_rule \
        --rule_path <stage-1 predictions.jsonl> --vendor tamu --seed 42 --force
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from llms.language_models.base_language_model import BaseLanguageModel
from qa_prediction.predict_answer import main as predict_answer_main

from llm_client import ChatMessage, LLMChatClient  # noqa: E402  (ToG's client, reused)
from llm_config import resolve_llm_config  # noqa: E402

# RoG sizes prompts against a 4096-token Llama-2 window. Every vendor we target
# is far larger, but keeping the upstream budget is the point: it holds the
# prompt (and so the shuffle/truncate behaviour) identical to the local engine.
MAXIMUN_TOKEN = 4096


class ApiLLM(BaseLanguageModel):
    """BaseLanguageModel backed by an OpenAI-compatible chat endpoint."""

    @staticmethod
    def add_args(parser):
        parser.add_argument("--vendor", default="tamu")
        parser.add_argument("--model", default=None,
                            help="override the vendor's default model")
        parser.add_argument("--timeout-s", type=float, default=120.0)
        parser.add_argument("--max_new_tokens", type=int, default=512)
        parser.add_argument("--seed", type=int, default=42,
                            help="seed for RoG's path shuffle (build_qa_input.py)")
        return parser

    def __init__(self, args):
        super().__init__(args)
        self.config = resolve_llm_config(vendor=args.vendor, model=args.model)
        self.client = LLMChatClient.from_connection_config(
            self.config, timeout_s=args.timeout_s
        )
        self.maximun_token = MAXIMUN_TOKEN  # upstream's spelling; do not "fix"

    def load_model(self, **kwargs):
        return None  # nothing to load: the model is remote

    def prepare_for_inference(self, **model_kwargs):
        print(f"reasoner LLM: {self.config.vendor}/{self.config.model}")

    def tokenize(self, text):
        """Approximate token count for RoG's prompt-length budgeting.

        RoG only uses this to decide when to drop paths, and it compares against
        maximun_token. A ~4 chars/token estimate keeps that decision close to the
        Llama tokenizer's without shipping a tokenizer or paying an API call.
        """
        return len(text) // 4

    def generate_sentence(self, llm_input):
        """Return the model's answer text, or None to let RoG skip the record."""
        try:
            return self.client.complete_json(
                [ChatMessage(role="user", content=llm_input)], temperature=0.0
            )
        except Exception as exc:  # upstream treats None as "no prediction"
            print(f"  [warn] LLM call failed, skipping record: {exc}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="rmanluo")
    parser.add_argument("--d", "-d", default="RoG-webqsp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--predict_path", default="results/KGQA")
    parser.add_argument("--model_name", default="RoG", help="model_name for save results")
    parser.add_argument("--prompt_path", default="prompts/llama2_predict.txt")
    parser.add_argument("--add_rule", action="store_true")
    parser.add_argument("--use_true", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--use_random", action="store_true")
    parser.add_argument("--each_line", action="store_true")
    parser.add_argument("--rule_path", default=None)
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("-n", type=int, default=1, help="number of processes")
    parser.add_argument("--filter_empty", action="store_true")
    parser.add_argument("--debug", action="store_true")
    ApiLLM.add_args(parser)
    args = parser.parse_args()

    # Must happen before main(): PromptBuilder shuffles at prompt-build time.
    # -n > 1 forks workers that would each inherit this seed and shuffle
    # identically per record, so determinism survives, but upstream's Pool
    # ordering does not affect the seeded draw either way.
    random.seed(args.seed)

    predict_answer_main(args, ApiLLM)
