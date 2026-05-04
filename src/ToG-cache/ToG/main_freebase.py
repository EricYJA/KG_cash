from tqdm import tqdm
import argparse
from utils import *
import random
from client import *
import os
import json
from trace_utils import TraceRecorder, set_active_trace_recorder


def load_completed_questions(output_path):
    completed_questions = set()
    if not output_path or not os.path.exists(output_path):
        return completed_questions
    with open(output_path, "r", encoding="utf-8") as infile:
        for line in infile:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            question = payload.get("question")
            if isinstance(question, str) and question:
                completed_questions.add(question)
    return completed_questions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=1024, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=1, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=1, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-5-mini-2025-08-07", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--test-limit", type=int,
                        default=400, help="only run the first k dataset samples.")
    parser.add_argument("--output-file", type=str,
                        default=None, help="path to save jsonl results. Defaults to ../output/ToG_<dataset>.jsonl.")
    parser.add_argument("--vendor", type=str,
                        default="openai", help="LLM vendor: tamu, openai, google. When set to 'tamu', uses the httpx-based client with LLM_API_KEY env var.")
    parser.add_argument("--trace-enabled", action="store_true", default=True,
                        help="enable per-question tracing for offline cache simulation.")
    parser.add_argument("--trace-output", type=str,
                        default=None, help="path to write question trace jsonl.")
    args = parser.parse_args()

    trace_output = args.trace_output or os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "traces", "tog_trace_{}.jsonl".format(args.dataset))
    recorder = TraceRecorder(
        enabled=args.trace_enabled,
        output_path=trace_output,
    )
    set_active_trace_recorder(recorder)

    datas, question_string = prepare_dataset(args.dataset)
    result_output_path = args.output_file or os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "ToG_{}.jsonl".format(args.dataset))
    completed_questions = load_completed_questions(result_output_path)
    if args.test_limit is not None:
        datas = datas[:min(args.test_limit, len(datas))]
    failed_questions = 0
    if completed_questions:
        print("Found %d completed questions in %s." % (len(completed_questions), result_output_path))

    try:
        for question_id, data in enumerate(tqdm(datas)):
            question = data[question_string]
            if question in completed_questions:
                continue
            topic_entity = data['topic_entity']
            recorder.start_question(
                question_id=question_id,
                dataset=args.dataset,
                question=question,
                question_field=question_string,
                initial_topic_entity=topic_entity,
            )
            final_status = "unknown"
            final_output = {}
            try:
                cluster_chain_of_entities = []
                pre_relations = [],
                pre_heads= [-1] * len(topic_entity)
                flag_printed = False
                for depth in range(1, args.depth+1):
                    recorder.set_depth(depth)
                    current_entity_relations_list = []
                    i=0
                    for entity in topic_entity:
                        if entity!="[FINISH_ID]":
                            retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)  # best entity triplet, entitiy_id
                            current_entity_relations_list.extend(retrieve_relations_with_scores)
                        i+=1
                    total_candidates = []
                    total_scores = []
                    total_relations = []
                    total_entities_id = []
                    total_topic_entities = []
                    total_head = []

                    for entity in current_entity_relations_list:
                        if entity['head']:
                            entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                        else:
                            entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                        
                        if len(entity_candidates_id) >=20:
                            original_candidates = list(entity_candidates_id)
                            entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)
                            recorder.record_event(
                                "entity_candidate_sampling",
                                input_payload={"entity": entity, "original_entity_ids": original_candidates, "threshold": 20, "retain_count": args.num_retain_entity},
                                output_payload={"sampled_entity_ids": entity_candidates_id},
                            )

                        if len(entity_candidates_id) ==0:
                            continue

                        scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)
                        
                        total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
                    
                    if len(total_candidates) ==0:
                        recorder.record_event("depth_end", input_payload={"depth": depth}, output_payload={"next_topic_entity": {}, "stopped": True, "reason": "no_total_candidates"})
                        half_stop(question, cluster_chain_of_entities, args)
                        final_status = "half_stop"
                        final_output = {"depth": depth, "reasoning_chains": cluster_chain_of_entities}
                        flag_printed = True
                        break
                        
                    flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
                    cluster_chain_of_entities.append(chain_of_entities)
                    if flag:
                        stop, results = reasoning(question, cluster_chain_of_entities, args)
                        if stop:
                            print("ToG stoped at depth %d." % depth)
                            recorder.record_event("depth_end", input_payload={"depth": depth}, output_payload={"next_topic_entity": {}, "stopped": True, "reason": "reasoning_stop"})
                            save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset, output_file=args.output_file)
                            final_status = "reasoning_stop"
                            final_output = {"depth": depth, "results": results, "reasoning_chains": cluster_chain_of_entities}
                            flag_printed = True
                            break
                        else:
                            print("depth %d still not find the answer." % depth)
                            topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                            recorder.record_event("depth_end", input_payload={"depth": depth}, output_payload={"next_topic_entity": topic_entity, "stopped": False})
                            continue
                    else:
                        recorder.record_event("depth_end", input_payload={"depth": depth}, output_payload={"next_topic_entity": {}, "stopped": True, "reason": "entity_prune_empty"})
                        half_stop(question, cluster_chain_of_entities, args)
                        final_status = "half_stop"
                        final_output = {"depth": depth, "reasoning_chains": cluster_chain_of_entities}
                        flag_printed = True
                        break
                
                if not flag_printed:
                    recorder.clear_depth()
                    results = generate_without_explored_paths(question, args)
                    save_2_jsonl(question, results, [], file_name=args.dataset, output_file=args.output_file)
                    final_status = "generated_without_paths"
                    final_output = {"results": results}
                completed_questions.add(question)
            except Exception as exc:
                final_status = "error"
                final_output = {"error": {"type": type(exc).__name__, "message": str(exc)}}
                failed_questions += 1
                print("Skipping question_id=%d due to %s: %s" % (question_id, type(exc).__name__, str(exc)))
            finally:
                recorder.clear_depth()
                recorder.finish_question(final_status=final_status, final_output_file=args.output_file or None, extra_output=final_output)
    finally:
        recorder.finalize_run()
        if failed_questions:
            print("Completed run with %d failed questions skipped." % failed_questions)
