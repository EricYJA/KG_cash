from __future__ import annotations

from collections import Counter

from .backend_adapter import KGBackendAdapter
from .config import LLMFrontendConfig
from .memory import PlannerMemory
from .planner import LLMPlanner
from .schemas import (
    FinalAnswerAction,
    InitialEntityAction,
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
        frontier: list[str] = []
        initial_frontier: list[str] = []
        observation = self.backend.describe_frontier(frontier)
        memory = PlannerMemory(
            max_steps=self.config.max_steps,
            max_history=self.config.max_memory_steps,
        )
        memory.set_frontier(frontier)

        query_counts: Counter[tuple[str, str, str]] = Counter()
        frontier_counts: Counter[str] = Counter(
            {memory.frontier_signature(frontier): 1}
        )
        query_trace: list[LLMQueryTraceStep] = []
        final_answers: list[str] = []
        initial_entity: str | None = None
        stop_reason = "step_limit"

        for _ in range(self.config.initial_entity_search_limit):
            initial_decision = self.planner.plan_next(
                example=example,
                memory=memory,
                observation=observation,
            )
            if initial_decision.error is not None or initial_decision.action is None:
                stop_reason = "invalid_model_output"
                break
            if not isinstance(initial_decision.action, InitialEntityAction):
                stop_reason = "missing_initial_entity"
                break

            initial_entity = initial_decision.action.entity
            frontier = self.backend.resolve_initial_frontier(initial_entity)
            if frontier:
                initial_frontier = frontier
                observation = self.backend.describe_frontier(frontier)
                memory.set_frontier(frontier)
                frontier_counts = Counter({memory.frontier_signature(frontier): 1})
                break

            memory.record_failed_initial_entity(initial_entity)
        else:
            stop_reason = "initial_entity_not_found"

        for step_id in range(1, self.config.max_steps + 1):
            if stop_reason != "step_limit":
                break

            decision = self.planner.plan_next(
                example=example,
                memory=memory,
                observation=observation,
            )
            if decision.error is not None or decision.action is None:
                stop_reason = "invalid_model_output"
                break

            action = decision.action
            if isinstance(action, InitialEntityAction):
                stop_reason = "unexpected_initial_entity"
                break

            if isinstance(action, FinalAnswerAction):
                final_answers = unique_strings(action.answers)[
                    : self.config.fallback_answer_limit
                ]
                if final_answers:
                    stop_reason = "final_answer"
                else:
                    final_answers = unique_strings(frontier)[
                        : self.config.fallback_answer_limit
                    ]
                    stop_reason = "final_answer_frontier_fallback"
                break

            query_key = (
                memory.frontier_signature(frontier),
                action.relation,
                action.direction,
            )
            query_counts[query_key] += 1
            if query_counts[query_key] > self.config.repeat_query_limit:
                stop_reason = "repeated_query_limit"
                break

            result = self.backend.execute_query(frontier, action)
            frontier = result.output_frontier
            observation = result.observation
            memory.record_query(step_id=step_id, action=action, result=result)
            query_trace.append(
                LLMQueryTraceStep(
                    step_id=step_id,
                    relation=action.relation,
                    direction=action.direction,
                    resolved_direction=result.resolved_direction,
                    output_frontier=result.output_frontier,
                )
            )

            frontier_signature = memory.frontier_signature(frontier)
            frontier_counts[frontier_signature] += 1
            if frontier_counts[frontier_signature] > self.config.repeat_frontier_limit:
                stop_reason = "repeated_frontier_limit"
                break

        if stop_reason != "final_answer" and not final_answers:
            final_answers = unique_strings(frontier)[
                : self.config.fallback_answer_limit
            ]

        return LLMRunTrace(
            question_id=example.question_id,
            question=example.question,
            llm_initial_entity=initial_entity,
            llm_initial_frontier=initial_frontier,
            llm_kg_queries=query_trace,
            llm_final_answer=final_answers,
            num_steps=len(query_trace),
            stop_reason=stop_reason,
        )
