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
import re
import sys
import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import qa_prediction.predict_answer as upstream_predict_answer
from llms.language_models.base_language_model import BaseLanguageModel
from qa_prediction.predict_answer import main as predict_answer_main

from llm_client import ChatMessage, LLMChatClient  # noqa: E402  (ToG's client, reused)
from llm_config import resolve_llm_config  # noqa: E402
from rog_e2e_metrics import (  # noqa: E402  (ToG's sidecar helpers, reused)
    append_question_metrics,
    metrics_sidecar_path,
)

# RoG sizes prompts against a 4096-token Llama-2 window. Every vendor we target
# is far larger, but keeping the upstream budget is the point: it holds the
# prompt (and so the shuffle/truncate behaviour) identical to the local engine.
MAXIMUN_TOKEN = 4096

# This prompt does two jobs.
#
# 1. FORMAT. RoG's evaluate_results.eval_f1 scores precision as
#    matched / len(answer.split("\n")) -- every output LINE counts as one predicted
#    entity. The fine-tuned RoG was trained to emit one bare entity per line, so that
#    denominator equals the number of answers. A chat model left to write prose
#    ("Based on the reasoning paths, ...") spreads one answer over many lines, which
#    inflates the denominator and craters precision even when the answer is right.
#    So: terse, one entity per line, no prose.
#
# 2. STRICT GROUNDING (no parametric fallback). The reasoner must answer ONLY from
#    the reasoning paths and output nothing when they don't contain the answer. This
#    is essential to the cache experiment, not a style choice: the experiment varies
#    which relation paths the cache serves and measures the accuracy delta. If the
#    model may answer from its own memory, accuracy stops depending on the paths --
#    every cache policy scores the same and the comparison becomes meaningless. A
#    strictly path-grounded reasoner makes accuracy a faithful function of path
#    quality, which is exactly what the cache is supposed to affect. It costs recall
#    (a path grounded to a raw Freebase MID yields no answer), but that lost recall
#    is a real property of the served paths, which is what we want to measure.
REASONER_SYSTEM_PROMPT = (
    "You extract answer entities from knowledge-graph reasoning paths. "
    "Output ONLY the answer(s), one entity name per line, copied verbatim from the "
    "reasoning paths. No numbering, no bullets, no markdown, no preamble, no "
    "explanation. If the reasoning paths do not contain the answer, output nothing."
)

# Lines that are framing rather than an answer entity. Anchored and specific so a
# real entity ("The Beatles", "Answer Man") is not dropped: only sentence lead-ins.
_FRAMING_RE = re.compile(
    r"^(based on|according to|here (are|is)\b|the answer\b|the following\b|"
    r"answer[:\s]|answers[:\s]|these are\b|from the reasoning|the reasoning|"
    r"no answer\b|not (contain|available)|there (is|are) no\b|note:)",
    re.IGNORECASE,
)


def _extract_answers(text):
    """Reduce a completion to bare answer entities, one per line.

    Conservative on purpose: it strips list/markdown decoration and drops obvious
    framing lines, but does not try to guess entities out of a paragraph. With the
    system prompt above the model already returns a clean list; this just repairs
    the occasional stray bullet or lead-in so precision is not taxed for it.
    """
    answers = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*•]\s+", "", line)      # bullet markers
        line = re.sub(r"^\d+[\.\)]\s+", "", line)          # "1." / "1)" numbering
        line = line.replace("**", "").replace("__", "").strip()
        if not line or line.endswith(":"):                 # empty or a lead-in
            continue
        if _FRAMING_RE.match(line):
            continue
        answers.append(line)
    return answers


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
        """Return newline-separated answer entities, or None to skip the record.

        The value is stored verbatim as `prediction`, and RoG's eval_f1 splits it
        on "\\n" to count predicted entities -- so returning terse one-per-line
        answers (not prose) is what keeps precision honest. See REASONER_SYSTEM_PROMPT.
        """
        try:
            text = self.client.complete_json(
                [
                    ChatMessage(role="system", content=REASONER_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=llm_input),
                ],
                temperature=0.0,
            )
        except Exception as exc:  # upstream treats None as "no prediction"
            print(f"  [warn] LLM call failed, skipping record: {exc}")
            return None
        return "\n".join(_extract_answers(text))


def install_stage2_timing():
    """Time each question in upstream's stage-2 loop into a ToG-shaped sidecar.

    Stage 2 is upstream's `predict_answer.main()`, run unmodified -- so the timer
    goes in by patching the two module globals it calls, not by editing it:

      `prediction(data, ...)` is upstream's per-question unit of work. It builds
      the prompt (which grounds stage 1's relation paths against the subgraph)
      and makes the reasoner LLM call, so bracketing it captures the whole of a
      question's stage-2 cost -- the part the planner cache never touches and
      that the old stage-1-only `full_speedup_x` pretended did not exist.

      `get_output_file(path, ...)` is where upstream reveals its output path.
      Wrapping it puts the sidecar beside predictions.jsonl without this file
      re-deriving the `rule_postfix` directory name, which would silently drift
      the moment upstream changed how that name is built.

    Both are looked up on the module at call time (including through the
    `partial` in the -n > 1 branch), so patching before `main()` covers both.
    """
    state = {"metrics_path": None}
    original_get_output_file = upstream_predict_answer.get_output_file
    original_prediction = upstream_predict_answer.prediction

    def get_output_file(path, force=False):
        state["metrics_path"] = metrics_sidecar_path(path)
        if force and state["metrics_path"] and os.path.exists(state["metrics_path"]):
            # --force rewrites predictions.jsonl from scratch; a stale sidecar
            # would leave the join pairing new stage-1 times with old stage-2 ones.
            os.remove(state["metrics_path"])
        return original_get_output_file(path, force=force)

    def prediction(data, processed_list, input_builder, model):
        # A question already in predictions.jsonl is upstream's resume skip: no
        # work happens, so timing it would enter a ~0s record and understate the
        # miss cost. Its record is already in the sidecar from the earlier run.
        if data["id"] in processed_list:
            return original_prediction(data, processed_list, input_builder, model)

        started = time.perf_counter()
        result = original_prediction(data, processed_list, input_builder, model)
        append_question_metrics(
            state["metrics_path"],
            {
                "id": data["id"],
                "elapsed_s": time.perf_counter() - started,
                # One reasoner call per question, whether or not it came back
                # usable; a failed call still cost its latency.
                "llm_calls": 1,
                "answered": result is not None,
            },
        )
        return result

    upstream_predict_answer.get_output_file = get_output_file
    upstream_predict_answer.prediction = prediction


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

    # Must also happen before main(): it patches the globals main() resolves.
    install_stage2_timing()

    predict_answer_main(args, ApiLLM)
