import argparse
import json
from pathlib import Path

from tqdm import tqdm

from freebase_function_rog import (
    generate_rog_answer,
    plan_relation_paths,
    retrieve_reasoning_paths,
    save_rog_jsonl,
)


DATASET_FILES = {
    "cwq": ("cwq.json", "question"),
    "webqsp": ("WebQSP.json", "RawQuestion"),
    "grailqa": ("grailqa.json", "question"),
    "simpleqa": ("SimpleQA.json", "question"),
    "qald": ("qald_10-en.json", "question"),
    "webquestions": ("WebQuestions.json", "question"),
    "trex": ("T-REX.json", "input"),
    "zeroshotre": ("Zero_Shot_RE.json", "input"),
    "creak": ("creak.json", "sentence"),
}


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def prepare_dataset(dataset_name):
    if dataset_name not in DATASET_FILES:
        raise ValueError(
            "dataset not found, you should pick from "
            "{cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}."
        )

    file_name, question_string = DATASET_FILES[dataset_name]
    data_path = _data_dir() / file_name
    if dataset_name == "grailqa" and not data_path.exists():
        data_path = _data_dir() / "graliqa.json"

    with open(data_path, encoding="utf-8") as f:
        datas = json.load(f)
    return datas, question_string


def extract_topic_entity(data):
    topic_entity = data.get("topic_entity")
    if isinstance(topic_entity, dict):
        return topic_entity

    # WebQSP raw records store topic entities under Parses rather than at top level.
    parses = data.get("Parses")
    if isinstance(parses, list):
        for parse in parses:
            entity_id = parse.get("TopicEntityMid")
            entity_name = parse.get("TopicEntityName") or parse.get("PotentialTopicEntityMention")
            if entity_id and entity_name:
                return {entity_id: entity_name}

    return {}


def run_sample(question, topic_entity, args):
    if topic_entity:
        relation_paths, raw_planner_output = plan_relation_paths(question, topic_entity, args)
        reasoning_paths = retrieve_reasoning_paths(topic_entity, relation_paths, args)
    else:
        relation_paths = []
        raw_planner_output = ""
        reasoning_paths = []

    answer = generate_rog_answer(question, reasoning_paths, args)
    save_rog_jsonl(
        question,
        answer,
        relation_paths,
        raw_planner_output,
        reasoning_paths,
        file_name=args.dataset,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cwq",
        help="choose the dataset.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="the max length of LLMs output.",
    )
    parser.add_argument(
        "--temperature_exploration",
        type=float,
        default=0.4,
        help="the temperature in exploration stage.",
    )
    parser.add_argument(
        "--temperature_reasoning",
        type=float,
        default=0,
        help="the temperature in reasoning stage.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=3,
        help="choose the search width of ToG.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="choose the search depth of ToG.",
    )
    parser.add_argument(
        "--remove_unnecessary_rel",
        type=bool,
        default=True,
        help="whether removing unnecessary relations.",
    )
    parser.add_argument(
        "--LLM_type",
        type=str,
        default="gpt-3.5-turbo",
        help="base LLM model.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="low",
        help="reasoning effort for GPT-5/o-series models: low, medium, or high.",
    )
    parser.add_argument(
        "--opeani_api_keys",
        type=str,
        default="",
        help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.",
    )
    parser.add_argument(
        "--num_retain_entity",
        type=int,
        default=5,
        help="Number of entities retained during entities search.",
    )
    parser.add_argument(
        "--prune_tools",
        type=str,
        default="llm",
        help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="only run the first k dataset samples.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="path to save jsonl results. Defaults to ../output/RoG_<dataset>.jsonl.",
    )
    parser.add_argument(
        "--vendor",
        type=str,
        default="tamu",
        help="LLM vendor: tamu, openai, google. When set to 'tamu', uses the httpx-based client with LLM_API_KEY env var.",
    )
    args = parser.parse_args()

    datas, question_string = prepare_dataset(args.dataset)
    if args.test_limit is not None:
        datas = datas[: min(args.test_limit, len(datas))]

    for data in tqdm(datas):
        question = data[question_string]
        topic_entity = extract_topic_entity(data)
        run_sample(question, topic_entity, args)
