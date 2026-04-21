from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .schemas import LLMRunTrace


def write_trace_jsonl(output_path: str | Path, traces: list[LLMRunTrace]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def summarize_traces(traces: list[LLMRunTrace]) -> dict[str, object]:
    stop_reason_counts = Counter(trace.stop_reason for trace in traces)
    avg_steps = sum(trace.num_steps for trace in traces) / len(traces) if traces else 0.0

    evaluated = [t for t in traces if t.gold_answers]
    hit1 = (
        sum(1 for t in evaluated if set(t.llm_final_answer) & set(t.gold_answers))
        / len(evaluated)
        if evaluated else None
    )

    per_question: dict[str, object] = {}
    for t in traces:
        entry: dict[str, object] = {"stop_reason": t.stop_reason}
        if t.gold_answers:
            entry["hit"] = bool(set(t.llm_final_answer) & set(t.gold_answers))
        per_question[t.question_id] = entry

    summary: dict[str, object] = {
        "examples": len(traces),
        "evaluated": len(evaluated),
        "avg_steps": avg_steps,
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
    }
    if hit1 is not None:
        summary["hit1"] = round(hit1, 4)
    summary["per_question"] = per_question
    summary["results"] = [trace.to_dict() for trace in traces]
    return summary
