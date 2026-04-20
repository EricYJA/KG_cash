from __future__ import annotations

import json
from pathlib import Path

from .schemas import QuestionExample, unique_strings


def resolve_webqsp_path(source: str | Path, split: str) -> Path:
    """Resolve a WebQSP file path from a dataset directory or direct file path."""

    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        return source_path / "data" / f"WebQSP.{split}.json"
    return source_path


def load_webqsp_examples(
    source: str | Path,
    split: str,
    limit: int | None = None,
) -> list[QuestionExample]:
    """Load a small QuestionExample view of the WebQSP dataset."""

    path = resolve_webqsp_path(source, split)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("Questions", [])
    examples: list[QuestionExample] = []
    for record in records:
        examples.append(_parse_question(record, split))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def _parse_question(record: dict[str, object], split: str) -> QuestionExample:
    parses = record.get("Parses")
    parse_list = parses if isinstance(parses, list) else []

    inferential_chain: list[str] = []
    gold_answers: list[str] = []

    for parse in parse_list:
        if not isinstance(parse, dict):
            continue
        if not inferential_chain:
            raw_chain = parse.get("InferentialChain")
            if isinstance(raw_chain, list):
                inferential_chain = [
                    str(relation_id).strip()
                    for relation_id in raw_chain
                    if str(relation_id).strip()
                ]
        answers = parse.get("Answers")
        if isinstance(answers, list):
            for answer in answers:
                if not isinstance(answer, dict):
                    continue
                answer_id = str(answer.get("AnswerArgument", "")).strip()
                if answer_id:
                    gold_answers.append(answer_id)

    return QuestionExample(
        question_id=str(record.get("QuestionId", "")).strip(),
        question=str(
            record.get("RawQuestion") or record.get("ProcessedQuestion") or ""
        ).strip(),
        gold_inferential_chain=inferential_chain,
        gold_answers=unique_strings(gold_answers),
        split=split,
    )
