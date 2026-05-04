import json
import os
import time
from contextlib import contextmanager

KG_TRACE_OPERATIONS = {
    "relation_lookup_head",
    "relation_lookup_tail",
    "entity_search",
    "entity_name_resolve",
}

LLM_TRACE_OPERATIONS = {
    "relation_prune_llm",
    "entity_score_llm",
    "reasoning_llm",
    "answer_generate_llm",
    "generate_without_explored_paths",
}

DROP_TRACE_OPERATIONS = {
    "depth_end",
    "half_stop",
    "save_result",
}


def now_ms():
    return time.time_ns() // 1_000_000


class TraceRecorder:
    def __init__(self, enabled=False, output_path=None):
        self.enabled = enabled
        self.output_path = output_path
        self.current_question = None
        self.current_depth = None
        self.current_question_started_at_ms = None
        self.current_question_started_at_perf_ns = None
        self.event_index = 0
        self.events = []
        self.completed_traces = []

    def start_question(self, question_id, dataset, question, question_field, initial_topic_entity):
        if not self.enabled:
            return
        self.current_question = {
            "question_id": question_id,
            "question": question,
            "initial_topic_entity": initial_topic_entity,
        }
        self.current_depth = None
        self.current_question_started_at_ms = now_ms()
        self.current_question_started_at_perf_ns = time.perf_counter_ns()
        self.event_index = 0
        self.events = []

    def set_depth(self, depth):
        self.current_depth = depth

    def clear_depth(self):
        self.current_depth = None

    def record_event(self, operation, input_payload=None, output_payload=None, metadata=None, status="ok", depth=None, error=None):
        if not self.enabled or self.current_question is None:
            return
        timestamp_ms = now_ms()
        event = {
            "event_index": self.event_index,
            "depth": self.current_depth if depth is None else depth,
            "operation": operation,
            "status": status,
            "start_time_ms": timestamp_ms,
            "end_time_ms": timestamp_ms,
            "duration_ms": 0,
            "input": input_payload or {},
            "output": output_payload or {},
        }
        if metadata:
            event["metadata"] = metadata
        if error:
            event["error"] = error
        self.events.append(event)
        self.event_index += 1

    @contextmanager
    def timed_event(self, operation, input_payload=None, metadata=None, depth=None):
        if not self.enabled or self.current_question is None:
            yield None
            return
        start_time_ms = now_ms()
        start_perf_ns = time.perf_counter_ns()
        event = {
            "event_index": self.event_index,
            "depth": self.current_depth if depth is None else depth,
            "operation": operation,
            "status": "ok",
            "start_time_ms": start_time_ms,
            "end_time_ms": start_time_ms,
            "duration_ms": 0,
            "input": input_payload or {},
            "output": {},
        }
        if metadata:
            event["metadata"] = metadata
        self.events.append(event)
        self.event_index += 1
        try:
            yield event
        except Exception as exc:
            event["status"] = "error"
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            end_time_ms = now_ms()
            end_perf_ns = time.perf_counter_ns()
            event["end_time_ms"] = end_time_ms
            event["duration_ms"] = max(0, (end_perf_ns - start_perf_ns) // 1_000_000)

    def finish_question(self, final_status, final_output_file=None, extra_output=None):
        if not self.enabled or self.current_question is None:
            return
        ended_at_ms = now_ms()
        duration_ms = max(0, (time.perf_counter_ns() - self.current_question_started_at_perf_ns) // 1_000_000)
        trace_obj = dict(self.current_question)
        trace_obj["started_at_ms"] = self.current_question_started_at_ms
        trace_obj["ended_at_ms"] = ended_at_ms
        trace_obj["duration_ms"] = duration_ms
        trace_obj["final_status"] = final_status
        trace_obj["events"] = [event for event in (self._sanitize_event(e) for e in self.events) if event is not None]
        self.completed_traces.append(trace_obj)
        self._write_trace(trace_obj)
        self.current_question = None
        self.current_depth = None
        self.current_question_started_at_ms = None
        self.current_question_started_at_perf_ns = None
        self.events = []
        self.event_index = 0

    def finalize_run(self):
        if not self.enabled or not self.output_path:
            return
        pretty_output_path = self._pretty_output_path()
        output_dir = os.path.dirname(pretty_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        all_traces = self._load_all_traces_from_jsonl()
        with open(pretty_output_path, "w", encoding="utf-8") as outfile:
            json.dump(all_traces, outfile, ensure_ascii=False, indent=2)
            outfile.write("\n")

    def _write_trace(self, trace_obj):
        output_path = self.output_path
        if not output_path:
            return
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(trace_obj, ensure_ascii=False) + "\n")

    def _pretty_output_path(self):
        if self.output_path.endswith(".jsonl"):
            return self.output_path[:-1]
        return self.output_path + ".pretty.json"

    def _load_all_traces_from_jsonl(self):
        traces = []
        if not self.output_path or not os.path.exists(self.output_path):
            return traces
        with open(self.output_path, "r", encoding="utf-8") as infile:
            for line in infile:
                stripped = line.strip()
                if not stripped:
                    continue
                traces.append(json.loads(stripped))
        return traces

    def _sanitize_event(self, event):
        if event.get("operation") in DROP_TRACE_OPERATIONS:
            return None
        sanitized = dict(event)
        if sanitized.get("status") == "ok":
            sanitized.pop("status", None)
        if sanitized.get("operation") not in KG_TRACE_OPERATIONS:
            sanitized.pop("input", None)
        sanitized.pop("metadata", None)
        sanitized.pop("output", None)
        sanitized["type"] = self._event_type(sanitized["operation"])
        return sanitized

    def _event_type(self, operation):
        if operation in KG_TRACE_OPERATIONS:
            return "KG"
        if operation in LLM_TRACE_OPERATIONS:
            return "LLM"
        return "OTHER"


_ACTIVE_TRACE_RECORDER = None


def set_active_trace_recorder(recorder):
    global _ACTIVE_TRACE_RECORDER
    _ACTIVE_TRACE_RECORDER = recorder


def get_active_trace_recorder():
    return _ACTIVE_TRACE_RECORDER


def is_trace_enabled():
    return _ACTIVE_TRACE_RECORDER is not None and _ACTIVE_TRACE_RECORDER.enabled
