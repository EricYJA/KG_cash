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
    return {
        "examples": len(traces),
        "avg_steps": avg_steps,
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
        "sample_results": [trace.to_dict() for trace in traces[:3]],
    }
