from __future__ import annotations

from collections import Counter

from .backend_adapter import KGBackendAdapter
from .config import LLMFrontendConfig
from .memory import PlannerMemory
from .planner import LLMPlanner
from .schemas import (
    ExploreAction,
    FinalAnswerAction,
    KGQueryAction,
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

        query_counts: Counter[tuple[str, str, str]] = Counter()
        frontier_counts: Counter[str] = Counter(
            {memory.frontier_signature(frontier): 1}
        )

        for step_id in range(1, self.config.max_steps + 1):
            # --- Phase 1: relation selection ---
            observation = self.backend.describe_frontier(frontier)
            frontier_labels = self.backend.get_entity_labels(frontier)
            rel_decision = self.planner.select_relation(
                example, memory, frontier, frontier_labels, observation
            )
            if rel_decision.error is not None or not isinstance(rel_decision.action, KGQueryAction):
                print(rel_decision)
                stop_reason = "invalid_model_output_relation"
                break

            action = rel_decision.action
            query_key = (
                memory.frontier_signature(frontier),
                action.relation,
                action.direction,
            )
            query_counts[query_key] += 1
            if query_counts[query_key] > self.config.repeat_query_limit:
                stop_reason = "repeated_query_limit"
                break

            input_frontier_snapshot = list(frontier)
            input_relations = (
                observation.forward_relations + observation.backward_relations
            )
            result = self.backend.execute_query(frontier, action)
            frontier = result.output_frontier
            memory.record_query(step_id=step_id, action=action, result=result)

            if not frontier:
                query_trace.append(LLMQueryTraceStep(
                    step_id=step_id,
                    relation=action.relation,
                    direction=action.direction,
                    resolved_direction=result.resolved_direction,
                    input_frontier=input_frontier_snapshot,
                    output_frontier=result.output_frontier,
                    input_relations=input_relations,
                    selected_relation=action.relation,
                ))
                stop_reason = "empty_frontier"
                break

            frontier_signature = memory.frontier_signature(frontier)
            frontier_counts[frontier_signature] += 1
            if frontier_counts[frontier_signature] > self.config.repeat_frontier_limit:
                query_trace.append(LLMQueryTraceStep(
                    step_id=step_id,
                    relation=action.relation,
                    direction=action.direction,
                    resolved_direction=result.resolved_direction,
                    input_frontier=input_frontier_snapshot,
                    output_frontier=result.output_frontier,
                    input_relations=input_relations,
                    selected_relation=action.relation,
                ))
                stop_reason = "repeated_frontier_limit"
                break

            # --- Phase 2: entity evaluation ---
            dest_labels = self.backend.get_entity_labels(frontier)
            eval_decision = self.planner.evaluate_entities(
                example, memory, frontier, dest_labels
            )
            if eval_decision.error is not None or eval_decision.action is None:
                print(eval_decision)
                query_trace.append(LLMQueryTraceStep(
                    step_id=step_id,
                    relation=action.relation,
                    direction=action.direction,
                    resolved_direction=result.resolved_direction,
                    input_frontier=input_frontier_snapshot,
                    output_frontier=result.output_frontier,
                    input_relations=input_relations,
                    selected_relation=action.relation,
                    input_entities=list(frontier),
                ))
                stop_reason = "invalid_model_output_entity"
                break

            if isinstance(eval_decision.action, FinalAnswerAction):
                final_answers = unique_strings(eval_decision.action.answers)[
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
                    relation=action.relation,
                    direction=action.direction,
                    resolved_direction=result.resolved_direction,
                    input_frontier=input_frontier_snapshot,
                    output_frontier=result.output_frontier,
                    input_relations=input_relations,
                    selected_relation=action.relation,
                    input_entities=list(frontier),
                    eval_action="FINAL_ANSWER",
                    eval_entities=final_answers,
                ))
                break

            # ExploreAction — narrow frontier to the single most promising entity
            assert isinstance(eval_decision.action, ExploreAction)
            query_trace.append(LLMQueryTraceStep(
                step_id=step_id,
                relation=action.relation,
                direction=action.direction,
                resolved_direction=result.resolved_direction,
                input_frontier=input_frontier_snapshot,
                output_frontier=result.output_frontier,
                input_relations=input_relations,
                selected_relation=action.relation,
                input_entities=list(frontier),
                eval_action="EXPLORE",
                eval_entities=[eval_decision.action.entity],
            ))
            frontier = [eval_decision.action.entity]
            memory.set_frontier(frontier)

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
