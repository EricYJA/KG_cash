from __future__ import annotations

from collections import Counter

from .backend_adapter import KGBackendAdapter
from .config import LLMFrontendConfig
from .memory import PlannerMemory
from .planner import LLMPlanner
from .schemas import (
    FinalAnswerAction,
    LLMQueryTraceStep,
    LLMRunTrace,
    QuestionExample,
    unique_strings,
)


class IterativeKGController:
    def __init__(
        self,
        planner: LLMPlanner,
        backend: KGBackendAdapter,
        config: LLMFrontendConfig,
    ) -> None:
        self.planner = planner
        self.backend = backend
        self.config = config

    def run(self, example: QuestionExample) -> LLMRunTrace:
        frontier = self.backend.initial_frontier(example.topic_entity)
        observation = self.backend.describe_frontier(frontier)
        memory = PlannerMemory(
            max_steps=self.config.max_steps,
            max_history=self.config.max_memory_steps,
        )
        memory.set_frontier(frontier)

        query_counts: Counter[tuple[str, str, str, str]] = Counter()
        frontier_counts: Counter[str] = Counter({memory.frontier_signature(frontier): 1})
        query_trace: list[LLMQueryTraceStep] = []
        final_answers: list[str] = []
        final_reason: str | None = None
        final_raw_output: str | None = None
        stop_reason = "step_limit"

        for step_id in range(1, self.config.max_steps + 1):
            decision = self.planner.plan_next(
                example=example,
                memory=memory,
                observation=observation,
            )
            final_raw_output = decision.raw_output
            if decision.error is not None or decision.action is None:
                stop_reason = "invalid_model_output"
                final_reason = decision.error
                break

            action = decision.action
            if isinstance(action, FinalAnswerAction):
                final_answers = unique_strings(action.answers)[: self.config.fallback_answer_limit]
                final_reason = action.reason
                if final_answers:
                    stop_reason = "final_answer"
                else:
                    final_answers = unique_strings(frontier)[: self.config.fallback_answer_limit]
                    stop_reason = "final_answer_frontier_fallback"
                    if final_reason is None:
                        final_reason = "Model returned FINAL_ANSWER without explicit answers."
                break

            if not frontier and not action.entity:
                stop_reason = "empty_frontier_without_entity"
                final_reason = action.reason
                break

            query_frontier = frontier if frontier else ([action.entity] if action.entity else [])
            query_key = (
                memory.frontier_signature(query_frontier),
                action.relation,
                action.direction,
                action.entity or "",
            )
            query_counts[query_key] += 1
            if query_counts[query_key] > self.config.repeat_query_limit:
                stop_reason = "repeated_query_limit"
                final_reason = action.reason
                break

            result = self.backend.execute_query(frontier, action)
            frontier = result.output_frontier
            observation = result.observation
            memory.record_query(step_id=step_id, action=action, result=result)
            query_trace.append(
                LLMQueryTraceStep(
                    step_id=step_id,
                    relation=action.relation,
                    direction=result.resolved_direction,
                    requested_direction=action.direction,
                    input_frontier=result.input_frontier,
                    output_frontier=result.output_frontier,
                    raw_model_output=decision.raw_output,
                    reason=action.reason,
                    frontier_after_hop=result.frontier_after_hop,
                    primitive_calls=result.primitive_calls,
                    cache_hits=result.cache_hits,
                    cache_misses=result.cache_misses,
                )
            )

            frontier_signature = memory.frontier_signature(frontier)
            frontier_counts[frontier_signature] += 1
            if frontier_counts[frontier_signature] > self.config.repeat_frontier_limit:
                stop_reason = "repeated_frontier_limit"
                final_reason = action.reason
                break

        if stop_reason != "final_answer" and not final_answers:
            final_answers = unique_strings(frontier)[: self.config.fallback_answer_limit]

        return LLMRunTrace(
            question_id=example.question_id,
            question=example.question,
            topic_entity=example.topic_entity,
            gold_inferential_chain=list(example.gold_inferential_chain),
            llm_kg_queries=query_trace,
            llm_final_answer=final_answers,
            gold_answers=list(example.gold_answers),
            num_steps=len(query_trace),
            stop_reason=stop_reason,
            llm_final_reason=final_reason,
            final_raw_model_output=final_raw_output,
            split=example.split,
        )
