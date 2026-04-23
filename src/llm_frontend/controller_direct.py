from __future__ import annotations

from collections import Counter

from .backend_adapter import KGBackendAdapter
from .config import LLMFrontendConfig
from .memory import PlannerMemory
from .planner import LLMPlanner
from .schemas import (
    ExploreMultiAction,
    FinalAnswerAction,
    LLMQueryTraceStep,
    LLMRunTrace,
    QuestionExample,
    unique_strings,
)


class DirectKGController:
    """Single-phase controller: LLM picks next entity directly from neighborhood."""

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
        memory = PlannerMemory(
            max_steps=self.config.max_steps,
            max_history=self.config.max_memory_steps,
        )
        query_trace: list[LLMQueryTraceStep] = []
        final_answers: list[str] = []
        initial_entity: str | None = None
        initial_frontier: list[str] = []
        frontier: list[str] = []
        stop_reason = "step_limit"

        # --- Phase 0: find initial entity ---
        for _ in range(self.config.initial_entity_search_limit):
            decision = self.planner.select_initial_entity(example, memory)
            if decision.error is not None or decision.action is None:
                stop_reason = "invalid_model_output_initial"
                break
            initial_entity = decision.action.entity
            frontier = self.backend.resolve_initial_frontier(initial_entity)
            if frontier:
                initial_frontier = frontier
                memory.set_frontier(frontier)
                break
            memory.record_failed_initial_entity(initial_entity)
        else:
            stop_reason = "initial_entity_not_found"

        if stop_reason != "step_limit":
            return self._make_trace(
                example, initial_entity, initial_frontier,
                query_trace, final_answers, stop_reason,
            )

        frontier_counts: Counter[str] = Counter(
            {memory.frontier_signature(frontier): 1}
        )

        for step_id in range(1, self.config.max_steps + 1):
            frontier_labels = self.backend.get_entity_labels(frontier)
            neighborhood = self.backend.get_neighborhood(frontier)

            if not neighborhood:
                stop_reason = "empty_frontier"
                break

            all_neighbors = unique_strings(
                [eid for _, neighbors in neighborhood for eid in neighbors]
            )
            neighbor_labels = self.backend.get_entity_labels(all_neighbors)

            decision = self.planner.select_next_entity(
                example, memory, frontier, frontier_labels, neighborhood, neighbor_labels
            )
            if decision.error is not None or decision.action is None:
                print(decision)
                stop_reason = "invalid_model_output"
                break

            if isinstance(decision.action, FinalAnswerAction):
                final_answers = unique_strings(decision.action.answers)[
                    : self.config.fallback_answer_limit
                ]
                if not final_answers:
                    final_answers = unique_strings(frontier)[
                        : self.config.fallback_answer_limit
                    ]
                    stop_reason = "final_answer_frontier_fallback"
                else:
                    stop_reason = "final_answer"
                query_trace.append(LLMQueryTraceStep(
                    step_id=step_id,
                    relation="(direct)",
                    direction="forward",
                    resolved_direction="forward",
                    input_frontier=list(frontier),
                    output_frontier=final_answers,
                    input_entities=all_neighbors,
                    eval_action="FINAL_ANSWER",
                    eval_entities=final_answers,
                ))
                break

            assert isinstance(decision.action, ExploreMultiAction)
            next_frontier = unique_strings(decision.action.entities)[: self.config.max_explore_entities]
            query_trace.append(LLMQueryTraceStep(
                step_id=step_id,
                relation="(direct)",
                direction="forward",
                resolved_direction="forward",
                input_frontier=list(frontier),
                output_frontier=next_frontier,
                input_entities=all_neighbors,
                eval_action="EXPLORE",
                eval_entities=next_frontier,
            ))

            frontier = next_frontier
            memory.set_frontier(frontier)

            frontier_signature = memory.frontier_signature(frontier)
            frontier_counts[frontier_signature] += 1
            if frontier_counts[frontier_signature] > self.config.repeat_frontier_limit:
                stop_reason = "repeated_frontier_limit"
                break

        if stop_reason != "final_answer" and stop_reason != "final_answer_frontier_fallback" and not final_answers:
            final_answers = unique_strings(frontier)[: self.config.fallback_answer_limit]

        return self._make_trace(
            example, initial_entity, initial_frontier,
            query_trace, final_answers, stop_reason,
        )

    def _make_trace(
        self,
        example: QuestionExample,
        initial_entity: str | None,
        initial_frontier: list[str],
        query_trace: list[LLMQueryTraceStep],
        final_answers: list[str],
        stop_reason: str,
    ) -> LLMRunTrace:
        return LLMRunTrace(
            question_id=example.question_id,
            question=example.question,
            llm_initial_entity=initial_entity,
            llm_initial_frontier=initial_frontier,
            llm_kg_queries=query_trace,
            llm_final_answer=final_answers,
            num_steps=len(query_trace),
            stop_reason=stop_reason,
            gold_answers=example.gold_answers,
        )
