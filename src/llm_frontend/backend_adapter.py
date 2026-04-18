from __future__ import annotations

from collections import Counter
from pathlib import Path

from kg_cache_backend import KGCache, KGStore, QueryExecutor

from .config import LLMFrontendConfig
from .schemas import BackendQueryResult, FrontierObservation, KGQueryAction, unique_strings


class KGBackendAdapter:
    def __init__(
        self,
        store: KGStore,
        cache: KGCache,
        executor: QueryExecutor,
        config: LLMFrontendConfig,
    ) -> None:
        self.store = store
        self.cache = cache
        self.executor = executor
        self.config = config

    @classmethod
    def from_path(
        cls,
        triples_path: str | Path,
        config: LLMFrontendConfig,
        cache_mode: str = "none",
        cache_capacity: int = 0,
    ) -> "KGBackendAdapter":
        store = KGStore.from_path(triples_path)
        cache = KGCache(store, mode=cache_mode, capacity=cache_capacity)
        executor = QueryExecutor(cache)
        return cls(store=store, cache=cache, executor=executor, config=config)

    def initial_frontier(self, topic_entity: str | None) -> list[str]:
        return self.executor.initial_frontier(topic_entity)

    def describe_frontier(self, frontier: list[str]) -> FrontierObservation:
        unique_frontier = unique_strings(frontier)
        sample_entities = unique_frontier[: self.config.max_frontier_entities]
        scan_entities = unique_frontier[: self.config.relation_scan_limit]

        forward_counts: Counter[str] = Counter()
        backward_counts: Counter[str] = Counter()
        for entity in scan_entities:
            for relation in self.cache.get_tail_relations(entity):
                forward_counts[relation] += 1
            for relation in self.cache.get_head_relations(entity):
                backward_counts[relation] += 1

        return FrontierObservation(
            frontier=unique_frontier,
            frontier_size=len(unique_frontier),
            sample_entities=sample_entities,
            forward_relations=[
                relation
                for relation, _ in forward_counts.most_common(self.config.max_relation_candidates)
            ],
            backward_relations=[
                relation
                for relation, _ in backward_counts.most_common(self.config.max_relation_candidates)
            ],
        )

    def execute_query(
        self,
        current_frontier: list[str],
        action: KGQueryAction,
    ) -> BackendQueryResult:
        input_frontier = unique_strings(current_frontier)
        if not input_frontier and action.entity:
            input_frontier = self.initial_frontier(action.entity)

        step_result = self.executor.execute_step(
            frontier=input_frontier,
            relation=action.relation,
        )
        resolved_direction = {
            "out": "forward",
            "in": "backward",
            "empty": "empty",
            "identity": "identity",
        }.get(step_result.direction, step_result.direction)
        observation = self.describe_frontier(step_result.frontier)
        return BackendQueryResult(
            relation=action.relation,
            requested_direction=action.direction,
            resolved_direction=resolved_direction,
            input_frontier=step_result.frontier_before,
            frontier_after_hop=step_result.frontier_after_hop,
            output_frontier=step_result.frontier,
            primitive_calls=step_result.primitive_calls,
            cache_hits=step_result.cache_hits,
            cache_misses=step_result.cache_misses,
            observation=observation,
        )
