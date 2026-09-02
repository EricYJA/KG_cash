import json
import os
import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from freebase_func import (
    abandon_rels,
    execurte_sparql,
    id2entity_name_or_type,
    replace_entities_prefix,
    replace_relation_prefix,
    sparql_head_entities_extract,
    sparql_head_relations,
    sparql_tail_entities_extract,
    sparql_tail_relations,
)
from utils import compute_bm25_similarity, retrieve_top_docs, run_llm


PATH_RE = re.compile(r"<PATH>(.*?)</PATH>", re.IGNORECASE | re.DOTALL)
RELATION_RE = re.compile(r"(?:[A-Za-z_][\w]*\.)+[A-Za-z_][\w]*")
RELATION_FULL_RE = re.compile(r"^(?:[A-Za-z_][\w]*\.)+[A-Za-z_][\w]*$")
ROG_PLAN_INSTRUCTION = (
    "Please generate valid Freebase relation paths that can be helpful for "
    "answering the question. Return each path as <PATH>relation.one<SEP>"
    "relation.two</PATH>. Use at most {depth} relations per path and return "
    "at most {width} paths."
)
ROG_ANSWER_INSTRUCTION_WITH_PATHS = (
    "Based on the reasoning paths, please answer the given question. Please "
    "keep the answer as simple as possible and return all possible answers as "
    "a list."
)
ROG_ANSWER_INSTRUCTION_NO_PATHS = (
    "Please answer the following question. Please keep the answer as simple as "
    "possible and return all possible answers as a list."
)


@dataclass(frozen=True)
class RetrievedStep:
    source_id: str
    source_name: str
    relation: str
    target_id: str
    target_name: str
    reverse: bool = False

    def as_tuple(self) -> Tuple[str, str, str]:
        return self.source_name, self.relation, self.target_name


def get_entity_name(entity_id: str, cache: Optional[Dict[str, str]] = None) -> str:
    if not entity_id:
        return "UnName_Entity"
    if entity_id == "[FINISH_ID]":
        return "[FINISH]"
    if cache is not None and entity_id in cache:
        return cache[entity_id]
    name = id2entity_name_or_type(entity_id)
    if cache is not None:
        cache[entity_id] = name
    return name


def get_relations_for_entity(entity_id: str, remove_unnecessary_rel: bool = True) -> Tuple[List[str], List[str]]:
    head_relations = replace_relation_prefix(
        execurte_sparql(sparql_head_relations % entity_id)
    )
    tail_relations = replace_relation_prefix(
        execurte_sparql(sparql_tail_relations % entity_id)
    )

    if remove_unnecessary_rel:
        head_relations = [rel for rel in head_relations if not abandon_rels(rel)]
        tail_relations = [rel for rel in tail_relations if not abandon_rels(rel)]

    return sorted(set(head_relations)), sorted(set(tail_relations))


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    results = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        results.append(item)
    return results


def _limit_entities(entity_ids: Sequence[str], limit: int) -> List[str]:
    entity_ids = _dedupe_preserve_order(entity_ids)
    if limit <= 0 or len(entity_ids) <= limit:
        return list(entity_ids)
    return random.sample(list(entity_ids), limit)


def _is_freebase_entity(entity_id: str) -> bool:
    return entity_id.startswith("m.")


def normalize_relation(relation: str) -> Optional[str]:
    relation = relation.strip()
    relation = relation.removeprefix("ns:")
    relation = relation.removeprefix("http://rdf.freebase.com/ns/")
    relation = relation.strip("`'\"{}[](),;")
    if RELATION_FULL_RE.fullmatch(relation):
        return relation
    return None


def _query_entity_ids(entity_id: str, relation: str, reverse: bool) -> List[str]:
    relation = normalize_relation(relation)
    if relation is None:
        return []
    if reverse:
        sparql = sparql_head_entities_extract % (relation, entity_id)
    else:
        sparql = sparql_tail_entities_extract % (entity_id, relation)
    entity_ids = replace_entities_prefix(execurte_sparql(sparql))
    return [candidate for candidate in entity_ids if _is_freebase_entity(candidate)]


def _format_relation_frontier(topic_entity: Dict[str, str], args) -> str:
    lines = []
    for entity_id, entity_name in topic_entity.items():
        if entity_id == "[FINISH_ID]":
            continue
        head_relations, tail_relations = get_relations_for_entity(
            entity_id,
            remove_unnecessary_rel=args.remove_unnecessary_rel,
        )
        relations = sorted(set(head_relations + tail_relations))
        if not relations:
            relation_text = "No available relations found."
        else:
            relation_text = "; ".join(relations)
        lines.append(f"- {entity_name} ({entity_id}): {relation_text}")
    return "\n".join(lines)


def build_planning_prompt(question: str, topic_entity: Dict[str, str], args) -> str:
    frontier = _format_relation_frontier(topic_entity, args)
    return (
        ROG_PLAN_INSTRUCTION.format(width=args.width, depth=args.depth)
        + "\n\nQuestion:\n"
        + question
        + "\n\nTopic entities and valid one-hop relation labels:\n"
        + frontier
        + "\n\nRelation paths:\n"
    )


def _parse_tagged_paths(text: str, depth: int) -> List[List[str]]:
    paths = []
    for match in PATH_RE.finditer(text):
        raw_path = match.group(1)
        relations = []
        for rel in raw_path.split("<SEP>"):
            rel = normalize_relation(rel)
            if rel:
                relations.append(rel)
        if not relations:
            relations = [
                rel
                for rel in (normalize_relation(rel) for rel in RELATION_RE.findall(raw_path))
                if rel
            ]
        if relations:
            paths.append(relations[:depth])
    return paths


def _parse_relation_tokens(text: str, depth: int) -> List[List[str]]:
    paths = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        relations = [
            rel
            for rel in (normalize_relation(rel) for rel in RELATION_RE.findall(line))
            if rel
        ]
        if relations:
            paths.append(relations[:depth])
    return paths


def parse_relation_paths(text: str, width: int, depth: int) -> List[List[str]]:
    paths = _parse_tagged_paths(text, depth)
    if not paths:
        paths = _parse_relation_tokens(text, depth)

    deduped = []
    seen = set()
    for path in paths:
        clean_path = tuple(
            rel for rel in (normalize_relation(rel) for rel in path) if rel
        )
        if not clean_path or clean_path in seen:
            continue
        seen.add(clean_path)
        deduped.append(list(clean_path))
        if len(deduped) >= width:
            break
    return deduped


def _get_topic_relation_pool(topic_entity: Dict[str, str], args) -> List[str]:
    relation_pool = []
    for entity_id in topic_entity:
        if entity_id == "[FINISH_ID]":
            continue
        head_relations, tail_relations = get_relations_for_entity(
            entity_id,
            remove_unnecessary_rel=args.remove_unnecessary_rel,
        )
        relation_pool.extend(head_relations)
        relation_pool.extend(tail_relations)
    return sorted(set(relation_pool))


def _bm25_plan_paths(question: str, topic_entity: Dict[str, str], args) -> List[List[str]]:
    relation_pool = _get_topic_relation_pool(topic_entity, args)
    if not relation_pool:
        return []
    relations, _ = compute_bm25_similarity(question, relation_pool, args.width)
    return [[relation] for relation in relations]


def _sentencebert_plan_paths(question: str, topic_entity: Dict[str, str], args) -> List[List[str]]:
    from sentence_transformers import SentenceTransformer

    relation_pool = _get_topic_relation_pool(topic_entity, args)
    if not relation_pool:
        return []
    model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b")
    relations, _ = retrieve_top_docs(question, relation_pool, model, args.width)
    return [[relation] for relation in relations]


def plan_relation_paths(question: str, topic_entity: Dict[str, str], args) -> Tuple[List[List[str]], str]:
    if args.prune_tools == "bm25":
        paths = _bm25_plan_paths(question, topic_entity, args)
        return paths, json.dumps({"planner": "bm25", "paths": paths})
    if args.prune_tools == "sentencebert":
        paths = _sentencebert_plan_paths(question, topic_entity, args)
        return paths, json.dumps({"planner": "sentencebert", "paths": paths})

    prompt = build_planning_prompt(question, topic_entity, args)
    raw_output = run_llm(
        prompt,
        args.temperature_exploration,
        args.max_length,
        args.opeani_api_keys,
        args.LLM_type,
        vendor=getattr(args, "vendor", None),
        model=getattr(args, "model", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
    )
    paths = parse_relation_paths(raw_output, args.width, args.depth)
    return paths, raw_output


def expand_one_relation(
    entity_id: str,
    entity_name: str,
    relation: str,
    name_cache: Dict[str, str],
    args,
) -> List[RetrievedStep]:
    steps = []
    for reverse in (False, True):
        candidate_ids = _query_entity_ids(entity_id, relation, reverse=reverse)
        candidate_ids = _limit_entities(candidate_ids, args.num_retain_entity)
        for candidate_id in candidate_ids:
            candidate_name = get_entity_name(candidate_id, name_cache)
            steps.append(
                RetrievedStep(
                    source_id=entity_id,
                    source_name=entity_name,
                    relation=relation,
                    target_id=candidate_id,
                    target_name=candidate_name,
                    reverse=reverse,
                )
            )
    return steps


def retrieve_paths_for_rule(
    start_entity_id: str,
    start_entity_name: str,
    relation_path: Sequence[str],
    args,
    name_cache: Dict[str, str],
) -> List[List[RetrievedStep]]:
    if not relation_path:
        return []

    frontier = deque([(start_entity_id, start_entity_name, [])])
    completed = []
    while frontier:
        entity_id, entity_name, current_path = frontier.popleft()
        depth = len(current_path)
        if depth == len(relation_path):
            completed.append(current_path)
            continue

        relation = relation_path[depth]
        next_steps = expand_one_relation(entity_id, entity_name, relation, name_cache, args)
        for step in next_steps:
            frontier.append((step.target_id, step.target_name, current_path + [step]))

    return completed


def retrieve_reasoning_paths(
    topic_entity: Dict[str, str],
    relation_paths: Sequence[Sequence[str]],
    args,
) -> List[List[RetrievedStep]]:
    name_cache = dict(topic_entity)
    reasoning_paths = []
    max_paths = max(args.width * max(args.num_retain_entity, 1), args.width)

    for relation_path in relation_paths:
        for entity_id, entity_name in topic_entity.items():
            if entity_id == "[FINISH_ID]":
                continue
            paths = retrieve_paths_for_rule(
                entity_id,
                entity_name,
                relation_path,
                args,
                name_cache,
            )
            reasoning_paths.extend(paths)
            if len(reasoning_paths) >= max_paths:
                return reasoning_paths[:max_paths]

    return reasoning_paths


def path_to_string(path: Sequence[RetrievedStep]) -> str:
    text = ""
    for index, step in enumerate(path):
        if index == 0:
            text += f"{step.source_name} -> {step.relation} -> {step.target_name}"
        else:
            text += f" -> {step.relation} -> {step.target_name}"
    return text.strip()


def reasoning_paths_to_chains(reasoning_paths: Sequence[Sequence[RetrievedStep]]) -> List[List[Tuple[str, str, str]]]:
    return [[step.as_tuple() for step in path] for path in reasoning_paths]


def build_reasoning_prompt(question: str, reasoning_paths: Sequence[Sequence[RetrievedStep]]) -> str:
    question = question.strip()
    if not question.endswith("?"):
        question += "?"

    if reasoning_paths:
        context = "\n".join(path_to_string(path) for path in reasoning_paths)
        return (
            ROG_ANSWER_INSTRUCTION_WITH_PATHS
            + "\n\nReasoning Paths:\n"
            + context
            + "\n\nQuestion:\n"
            + question
            + "\nAnswer:"
        )

    return (
        ROG_ANSWER_INSTRUCTION_NO_PATHS
        + "\n\nQuestion:\n"
        + question
        + "\nAnswer:"
    )


def generate_rog_answer(question: str, reasoning_paths: Sequence[Sequence[RetrievedStep]], args) -> str:
    prompt = build_reasoning_prompt(question, reasoning_paths)
    return run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.opeani_api_keys,
        args.LLM_type,
        vendor=getattr(args, "vendor", None),
        model=getattr(args, "model", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
    )


def default_output_path(dataset: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "output",
        f"RoG_{dataset}.jsonl",
    )


def save_rog_jsonl(
    question: str,
    answer: str,
    relation_paths: Sequence[Sequence[str]],
    raw_planner_output: str,
    reasoning_paths: Sequence[Sequence[RetrievedStep]],
    file_name: str,
    output_file: Optional[str] = None,
) -> None:
    output_path = output_file or default_output_path(file_name)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    record = {
        "question": question,
        "results": answer,
        "reasoning_chains": reasoning_paths_to_chains(reasoning_paths),
        "predicted_paths": [list(path) for path in relation_paths],
        "planner_output": raw_planner_output,
    }
    with open(output_path, "a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
