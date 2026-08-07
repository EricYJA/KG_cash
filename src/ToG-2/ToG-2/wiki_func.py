import time,json,random,heapq,math
from utils import *
from utils import _complete_openai_compatible
from search import *
from openai import OpenAI

def transform_relation(wiki_relation):
    relation_without_prefix = wiki_relation.replace("wiki.relation.", "").replace("_", " ")
    return relation_without_prefix

def clean_relations(string, entity_id, head_relations,args):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        wiki_relation = match.group("relation").strip()
        wiki_relation = transform_relation(wiki_relation)
        if ';' in wiki_relation:
            continue
        score = match.group("score")
        if not wiki_relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if wiki_relation in head_relations:
            relations.append({"entity_id": entity_id, "relation": wiki_relation, "score": score, "head": True})
        else:
            relations.append({"entity_id": entity_id, "relation": wiki_relation, "score": score, "head": False})

    if not relations:
        return False, "No relations found"
    filtered_relations = [x for x in relations if x['score'] >= 0.2]
    if not filtered_relations:
        return False, "No relations found"
    sorted_data = sorted(filtered_relations, key=lambda x: x['score'], reverse=True)[0:args.width]
    return True, sorted_data

def clean_relation_all_e(results, all_entity_relations):
    entities_info = []
    entity_sections = re.split(r"Entity\s?\d+:", results)[1:]
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    for section in entity_sections:
        section = section.strip()
        try:
            entity_name = re.match(r"(.+?)\n(.+)", section, re.DOTALL).group(1)
            entity_name = entity_name.strip()
        except AttributeError:
            print("=======================No entity name found", section)
            continue
        if entity_name not in all_entity_relations.keys():
            print("Matching error for entity name", entity_name, all_entity_relations.keys())
            continue
        for match in re.finditer(pattern, section):
            match_relation = match.group("relation").strip().lower()
            match_relation = transform_relation(match_relation)
            if ';' in match_relation:
                continue
            score = match.group("score")
            temp = {}
            temp["relation"] = match_relation
            temp["score"] = float(score)
            relation = [d for d in all_entity_relations[entity_name] if d["relation"] == temp["relation"]]
            if not relation:
                print("Matching error for relation", match_relation, all_entity_relations[entity_name])
                continue

            temp["entity_id"] = relation[0]["entity_id"]
            temp["entity_name"] = entity_name
            temp['head'] = relation[0]["head"]
            entities_info.append(temp)

    if not entities_info:
        return False, "No relations found"

    seen = set()
    temp_list = []
    for item in entities_info:
        entity_id = item['entity_id']
        relation = item['relation']
        if (entity_id, relation) in seen:
            continue
        seen.add((entity_id, relation))
        temp_list.append(item)
    return True, temp_list

def construct_all_relation_prune_prompt(question, all_entity_relations, args):
    temp_prompt = extract_all_relation_prompt_wiki % (args.width, args.width, args.width) + question
    for i, entity_relations in enumerate(all_entity_relations.values(), start=1):
        if len(entity_relations) > 0:
            temp_prompt += ('\nEntity %s: ' % i + entity_relations[0]['entity_name'] + '\nAvailable Relations:\n' +
                            '\n'.join([f"{i}. {item['relation']}" for i, item in enumerate(entity_relations, start=1)]))
    return temp_prompt + '\nAnswer:'

def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt_wiki % (args.width)+question+'\nTopic Entity: '+entity_name+ '\nRelations:\n'+'\n'.join([f"{i}. {item}" for i, item in enumerate(total_relations, start=1)])+'Answer:\n'


def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)


def abandon_rels(relation):
    useless_relation_list = ["category's main topic", "topic\'s main category", "stack exchange site", 'main subject', 'country of citizenship', "commons category", "commons gallery", "country of origin", "country", "nationality"]
    if check_end_word(relation) or 'wikidata' in relation.lower() or 'wikimedia' in relation.lower() or relation.lower() in useless_relation_list:
        return True
    return False


def construct_entity_score_prompt0(question, relation, entity_candidates):
    return score_entity_candidates_prompt_wiki.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def construct_entity_find_prompt(question,topic_entity_name, relation, entity_candidates):
    
    Triplet='('+'{'+topic_entity_name+'}'+'--'+'{'+relation+'}'+'--'+'?'+')'

    entities=[]
    entity_pack=[]
    for entity in entity_candidates:

        text_list = [d['text'] for d in entity['related_sentences']]

        doc=''.join(text_list)

        entity_pack.append('{Entity: '+entity['name']+'}: {Reference: '+doc[0:300]+'}')

        entities.append('{'+entity['name']+'}')
        
    Entites=",".join(entities)

    check_prompt = find_entity_candidates_prompt_wiki2.format(question, Triplet,Entites) + "; \n".join(entity_pack) + '\nA: '
    final_prompt = find_entity_candidates_prompt_wiki + check_prompt

    return final_prompt,check_prompt

def relation_prune_all(all_entity_relations, question, args):
    prompt = construct_all_relation_prune_prompt(question, all_entity_relations, args)
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type_rp, vendor=getattr(args, "vendor", None), reasoning_effort=getattr(args, "reasoning_effort", None))
    flag, retrieve_relations_with_scores = clean_relation_all_e(result, all_entity_relations)
    if flag:
        return retrieve_relations_with_scores
    else:
        return []

def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, wiki_client):
    pre_relations = [pre_relations]
    relations = wiki_client.query_all("get_all_relations_of_an_entity", entity_id)
    head_relations = [rel['label'] for rel in relations['head']]
    tail_relations = [rel['label'] for rel in relations['tail']]
    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    if pre_head:
        tail_relations = set(tail_relations) - set(pre_relations)
        head_relations = set(head_relations)
    else:
        head_relations = set(head_relations) - set(pre_relations)

    total_relations = list(head_relations|tail_relations)
    total_relations.sort()

    prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), reasoning_effort=getattr(args, "reasoning_effort", None))
    flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations,args)
    if flag:
        return retrieve_relations_with_scores
    else:
        return []

def relation_search_prune_fin(entity_name, question, pre_triples, args):

    prompt = """
    ### 你的任务是：给定一个基于知识图谱推理的问题，和当前探索到的三元组链路，从给定的关系中选择最适合下一步应探索的关系路径。
    ### 给你举两个例子说明:
    ### 示例1
    [输入]:
    问题: 重庆啤酒进行过大宗交易的上市企业哪些业绩下降了？
    三元组链路: 重庆啤酒
    当前实体: 重庆啤酒
    关系:
        1.大宗交易
        2.主营
        3.子公司
        4.兄弟公司
        5.下属公司
        6.供应商
        7.客户
        （其中客户、供应商属于对立关系）
    [输出]：
    {{
    \"分析\" : \"当前三元组链路只有一个实体，是问题推理的起始阶段，根据问题需求，需要寻找和重庆啤酒进行过<大宗交易>的实体，而给定的关系中有<大宗交易>。因此，首先要探索<大宗交易>。\",
    \"关系\" : [\"大宗交易\"]
    }}
    
    ### 示例2
    [输入]:
    问题: 2022年祥鑫科技的哪些竞争对手出现了严重法律风险和严重资金风险？
    三元组链路: 祥鑫科技
    当前实体: 祥鑫科技
    关系:
        1.大宗交易
        2.主营
        3.子公司
        4.兄弟公司
        5.下属公司
        6.供应商
        7.客户
        （其中客户、供应商属于对立关系）
    [输出]：
    {{
    \"分析\" : \"当前三元组链路只有一个实体，是问题推理的起始阶段，根据问题需求，需要首先检索出与祥鑫科技有竞争关系的实体。而给定关系中没有直接的竞争关系，因此考虑通过多跳关系(如探索和祥鑫科技有相同<客户>的实体，或者有相同<供应商>的实体来寻找有竞争关系的实体。因此，首先要探索<客户>和<供应商>。\",
    \"关系\" : [\"客户\",\"供应商\"]
    }}
        
    ### 示例3
    [输入]:
    问题: ST八菱(八菱科技)有哪些潜在竞争对手也是一汽的供应商？
    三元组链路: 八菱科技-供应商-银邦股份 
    当前实体: 银邦股份
    关系:
        1.大宗交易
        2.主营
        3.子公司
        4.兄弟公司
        5.下属公司
        6.供应商
        7.客户
        （其中客户、供应商属于对立关系）
    [输出]：
    {{
    \"分析\" : \"根据问题需求，需要首先检索出与八菱科技有竞争关系的实体。而给定关系中没有直接的<竞争>关系，因此考虑通过多跳关系(如探索和八菱科技有相同<客户>的实体，或者有相同<供应商>的实体)来寻找有竞争关系的实体。当前三元组链路探索到了八菱科技的供应商是银邦股份。因此，接下来要探索哪些实体的供应商和八菱科技一样，也是银邦股份。即需要探索银邦股份的<客户>。\",
    \"关系\" : [\"客户\"]
    }}

    ### 请根据上面的指导仔细思考下面的问题：
    ### 注意，请严格参考示例的输出格式，以json形式输出你的回答。
    [输入]:
    问题: {}
    三元组链路: {}
    当前实体: {}
    关系:
        1.大宗交易
        2.主营
        3.子公司
        4.兄弟公司
        5.下属公司
        6.供应商
        7.客户
        （其中客户、供应商属于对立关系）
    [输出]：
    """
    max_attempts = 3
    prompt = prompt.format(question, pre_triples, entity_name)
    for attempt in range(max_attempts):
        result = run_llm_json(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys,args,
                              args.LLM_type)
        print(result,type(result))
        try:
            result = json.loads(result)
        except json.decoder.JSONDecodeError:
            if attempt == max_attempts - 1:
                raise ValueError("尝试了 3 次，仍未获取到有效的'关系'字段")
            else:
                continue

        if "关系" in result and len(result["关系"]) != 0:
            break

    return result


def relation_search(entity_id, entity_name, pre_relations, pre_head, question, args, wiki_client):
    relations = wiki_client.query_all("get_all_relations_of_an_entity", entity_id)
    head_relations = [rel['label'].lower() for rel in relations['head']]
    tail_relations = [rel['label'].lower() for rel in relations['tail']]
    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    if pre_head:
        tail_relations = set(tail_relations) - set(pre_relations)
    else:
        head_relations = set(head_relations) - set(pre_relations)

    head_relations = list(set(head_relations))
    h = [{"relation": s, 'head': True, 'entity_name': entity_name, 'entity_id': entity_id} for s in head_relations]
    tail_relations = list(set(tail_relations))
    t = [{"relation": s, 'head': False, 'entity_name': entity_name, 'entity_id': entity_id} for s in tail_relations]
    total_relations = h + t
    return total_relations

def run_llm_json(prompt, temperature, max_tokens, openai_api_keys, args, engine="gpt-3.5-turbo"):
    response_format = {"type": "text"} if "llama" in engine.lower() else {"type": "json_object"}
    result = _complete_openai_compatible(
        prompt,
        temperature,
        max_tokens,
        openai_api_keys,
        engine,
        1,
        getattr(args, "vendor", None),
        getattr(args, "reasoning_effort", None),
        sys_prompt="You are a helpful assistant designed to output JSON.",
        response_format=response_format,
    )
    return result

def topic_e_prune(question, entities, args):
    def extract_output(text):

        match = re.search(r'Output:\s*(\{.*?\})', text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return ""
    def construct_topic_prune_prompt(question, entities):
        entities_json_string = json.dumps(entities)
        prompt = 'question: ' + question + '\ntopic entities:\n' + entities_json_string + '\nOutput:'
        return prompt


    prompt = construct_topic_prune_prompt(question, entities)
    prompt = topic_prune_demos + '\n' + prompt
    results = run_llm_json(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args,args.LLM_type)
    if 'llama' in args.LLM_type:
        json_str = extract_output(results).strip()
        try:
            results = json.loads(json_str)
        except ValueError:
            print("Entity prune failed, output original entities")
            return entities
    else:
        try:
            results = json.loads(results)
        except Exception as e:
            print("Entity prune failed, output original entities. Error result:{}. Error Info{}".format(results, e))
            return entities

    print('Select:\n' + ' '.join(results))
    return results


def all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def entity_search(entity_id, relation, wiki_client, head):
    rid = wiki_client.query_all("label2pid", relation)
    if not rid or rid == "Not Found!":
        return []

    rid_str = rid.pop()

    entities = wiki_client.query_all("get_tail_entities_given_head_and_relation", entity_id, rid_str)

    if head:
        entities_set = entities['tail']
    else:
        entities_set = entities['head']

    if not entities_set:
        values = wiki_client.query_all("get_tail_values_given_head_and_relation", entity_id, rid_str)
        candidate_list = [{'name': name, 'id': '[FINISH_ID]'} for name in list(values)]
    else:
        candidate_list = []
        for item in entities_set:
            if item['label'] != "N/A":
                find_entity_name = item['label']
                find_entity_id = item['qid']
                candidate_list.append({'name': find_entity_name, 'id': find_entity_id})

    if len(candidate_list) >= 50:
        candidate_list = random.sample(candidate_list, 50)

    return candidate_list


def entity_search_fin(entity_id, relation, db_client, except_entities):
    print("entity search for {}".format(relation['pre_path']))
    if relation['relation'] not in ['客户', "供应商","子公司", "下属公司","主营"]:
        entities = db_client.get_tail_nodes_by_relation(entity_id, relation['relation'])
    elif relation['relation'] in ['客户', "供应商"]:
        entities = db_client.get_tail_nodes(entity_id, '主语', relation["relation"])
        if  relation['relation'] == "客户":
            entities.extend(db_client.get_tail_nodes(entity_id, '宾语', "供应商"))
        else:
            entities.extend(db_client.get_tail_nodes(entity_id, '宾语', "客户"))
    else:
        entities = db_client.get_tail_nodes(entity_id, '主语', relation["relation"])

    if not entities:
        print("not found tail entity of {} {}".format(entity_id,relation["relation"]))
        candidate_list=[{'name':'no entity','id':'[FINISH_ID]', 'pre_path':relation['pre_path']}]
    else:
        candidate_list=[]
        for item in entities:
            if item and item[0].startswith('机构:'):
                # print("find {}".format(item))
                find_entity_name = item[1]
                find_entity_id = item[0]
                candidate_list.append({'name':find_entity_name,'id':find_entity_id, 'pre_path':relation['pre_path']})
            else:
                candidate_list.append({'name':'no entity' ,'id':'[FINISH_ID]', 'pre_path':relation['pre_path']})
    candidate_list = [candidate for candidate in candidate_list if not any(candidate['id'] == except_entity['id'] for except_entity in except_entities)]
    if len(candidate_list) >= 50:
        candidate_list = random.sample(candidate_list, 50)
    
    return candidate_list


def entity_find(question, entity_candidates, topic_entity_name, relation, args):
    if len(entity_candidates) < 3:
        print('entity_candidates<3')
        return entity_candidates
    if len(entity_candidates) == 0:
        return []

    prompt , check_prompt= construct_entity_find_prompt(question, topic_entity_name, relation, entity_candidates)
    
    print('-----------entity_find check_prompt-----------')
    print(check_prompt)
    print('-----------entity_find end check_prompt-----------')

    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), reasoning_effort=getattr(args, "reasoning_effort", None))
    
    print('-----------entity_find result-----------')
    print(result)

    pattern = r'\{([^}]*)\}'
    find_re = re.findall(pattern, result)

    top2_entities=[]
    for string in find_re:
        if ',' in string:
            result = string.split(',')
            top2_entities.extend(result)
        else:
            top2_entities.append(string)

    print('-----------entity_find-----------')
    print(top2_entities)

    entity_candidates_find = match_top2_entities(top2_entities,entity_candidates)

    return entity_candidates_find

def contains_yes_regex(text):
    words = re.findall(r'\b\w+\b', text.lower())

    first_100_words = words[:100]
    if 'yes' in first_100_words:
        return True
    else:
        return False
def match_top2_entities(top2_entities,entity_candidates):
    entity_candidates_find=[]

    for entity in top2_entities:
        flag=False
        for entity_candidate in entity_candidates:
            if entity==entity_candidate['name']:
                flag=True
                entity_candidates_find.append(entity_candidate)
        if not flag:
            print('top2_entities not match entity_candidate[name]')
            print('entity_candidate[name]: '+','.join([d['name'] for d in entity_candidates]))
            
    print('\n---------------check top2_entities match back entity_candidates_name ---------------')

    print(','.join([d['name'] for d in entity_candidates_find]))

    return entity_candidates_find

      
def update_history_find_entity(entity_candidates_find,relation, total_candidates):

    for entity_candidate in entity_candidates_find:
        candidate={}
        candidate['relation'] = relation['relation']
        candidate['topic_entities'] = relation['entity_name']
        if 'head' in relation:
            candidate['head'] = relation['head']
        candidate['id'] = entity_candidate['id']
        candidate['name'] = entity_candidate['name']
        candidate['related_paragraphs'] = entity_candidate['related_paragraphs']
        if 'pre_path' in entity_candidate:
            candidate["pre_path"] = entity_candidate['pre_path']

        total_candidates.append(candidate)

    return total_candidates



def para_rank_topk_fin(question, Indepth_total_candidates, args, emb_model,k = 10):

    paras = []
    for candidate in Indepth_total_candidates:
        paras.extend(candidate['related_paragraphs'])
    scores = s2p_relevance_scores(paras, question, args, emb_model)
    top_paragraphs_heap = []
    cnt = 0
    for candidate in Indepth_total_candidates:
        for paragraph in candidate['related_paragraphs']:
            heapq.heappush(top_paragraphs_heap, (scores[cnt], paragraph, candidate['name'], candidate['id'], candidate['relation'], candidate['topic_entities'], candidate['pre_path']))
            cnt += 1
            if len(top_paragraphs_heap) > k:
                heapq.heappop(top_paragraphs_heap)
    top_k_paragraphs = []
    while top_paragraphs_heap:
        score, text, e_name, e_id, rel, topic_e, pp = heapq.heappop(top_paragraphs_heap)
        top_k_paragraphs.append({'entity_name': e_name, 'paragraph': text, 'score': score, 'entity_id': e_id, 'relation': rel, 'topic_entitie': topic_e, 'pre_path': pp})

    top_k_paragraphs.reverse()
    entities_with_score = {}
    alpha = 0.8
    for rank, paragraph in enumerate(top_k_paragraphs, start=1):

        score = float(paragraph['score'])
        weight = math.exp(-alpha * rank)
        name_idx = paragraph['entity_name']
        if name_idx in entities_with_score:
            entities_with_score[name_idx] += score * weight
        else:
            entities_with_score[name_idx] = score * weight

    sorted_entities = sorted(entities_with_score.items(), key=lambda x: x[1], reverse=True)
    sorted_entity_list = []

    for i in range(min(args.width, len(sorted_entities))):
        ename = sorted_entities[i][0]
        score = sorted_entities[i][1]
        flg = 0
        sents = []
        # paras = []
        for j, paragraph in enumerate(top_k_paragraphs):
            if ename == paragraph['entity_name']:
                if flg == 0:
                    eid = paragraph['entity_id']
                    tpc_e = paragraph['topic_entitie']
                    rel = paragraph['relation']
                    pp = paragraph['pre_path'] + '-' + ename
                    flg = 1
                splited_sentences = split_sentences(paragraph['paragraph'])
                sents.extend(splited_sentences)
        sents_dict = [{'text': s, 'score': 0} for s in sents]
        entity_info = {
            'id': eid,
            'name': ename,
            'entity_score': score,
            'topic_entities': tpc_e,
            'relation': rel,
            'pre_path': pp
        }
        Indepth_total_candidates.append(entity_info)
        entity_info['sentences'] = sents_dict
        sorted_entity_list.append(entity_info)


    entities_id = [d['id'] for d in sorted_entity_list]
    relations = [d['relation'] for d in sorted_entity_list]
    entities_name = [d['name'] for d in sorted_entity_list]
    topics = [d['topic_entities'] for d in sorted_entity_list]
    prep = [d['pre_path'] for d in sorted_entity_list]


    if len(entities_id) ==0:
        return False, [], [], [], [],[],[]

    cluster_chain_of_entities = [(topics[i], relations[i], entities_name[i]) for i in range(len(entities_name))]

    return True, cluster_chain_of_entities, entities_id, relations, prep, sorted_entity_list, Indepth_total_candidates



def para_rank_topk(question, Indepth_total_candidates, args, emb_model,k = 10):

    paras = []
    for candidate in Indepth_total_candidates:
        paras.extend(candidate['related_paragraphs'])
    scores = s2p_relevance_scores(paras, question, args, emb_model)
    top_paragraphs_heap = []
    cnt = 0
    for candidate in Indepth_total_candidates:
        for paragraph in candidate['related_paragraphs']:
            heapq.heappush(top_paragraphs_heap, (scores[cnt], paragraph, candidate['name'], candidate['id'], candidate['relation'], candidate['topic_entities'], candidate['head']))
            cnt += 1
            if len(top_paragraphs_heap) > k:
                heapq.heappop(top_paragraphs_heap)
    top_k_paragraphs = []
    while top_paragraphs_heap:
        score, text, e_name, e_id, rel, topic_e, h, = heapq.heappop(top_paragraphs_heap)
        top_k_paragraphs.append({'entity_name': e_name, 'paragraph': text, 'score': score, 'entity_id': e_id, 'relation': rel, 'topic_entitie': topic_e, 'head': h})

    top_k_paragraphs.reverse()
    entities_with_score = {}
    alpha = 0.8
    for rank, paragraph in enumerate(top_k_paragraphs, start=1):

        score = float(paragraph['score'])
        weight = math.exp(-alpha * rank)
        name_idx = paragraph['entity_name']
        if name_idx in entities_with_score:
            entities_with_score[name_idx] += score * weight
        else:
            entities_with_score[name_idx] = score * weight

    sorted_entities = sorted(entities_with_score.items(), key=lambda x: x[1], reverse=True)
    sorted_entity_list = []

    for i in range(min(args.width, len(sorted_entities))):
        ename = sorted_entities[i][0]
        score = sorted_entities[i][1]
        flg = 0
        sents = []
        for j, paragraph in enumerate(top_k_paragraphs):
            if ename == paragraph['entity_name']:
                if flg == 0:
                    eid = paragraph['entity_id']
                    tpc_e = paragraph['topic_entitie']
                    rel = paragraph['relation']
                    h = paragraph['head']
                    flg = 1
                splited_sentences = split_sentences_windows(paragraph['paragraph'], *args.sliding_window)
                sents.extend(splited_sentences)
        sents_dict = [{'text': s, 'score': 0} for s in sents]

        entity_info = {
            'id': eid,
            'name': ename,
            'entity_score': score,
            'topic_entities': tpc_e,
            'relation': rel,
            'head': h
        }
        Indepth_total_candidates.append(entity_info)
        entity_info['sentences'] = sents_dict
        sorted_entity_list.append(entity_info)


    entities_id = [d['id'] for d in sorted_entity_list]
    relations = [d['relation'] for d in sorted_entity_list]
    entities_name = [d['name'] for d in sorted_entity_list]
    topics = [d['topic_entities'] for d in sorted_entity_list]
    heads = [d['head'] for d in sorted_entity_list]

    if len(entities_id) ==0:
        return False, [], [], [], [],[]

    cluster_chain_of_entities = [(topics[i], relations[i], entities_name[i]) for i in range(len(entities_name))]

    return True, cluster_chain_of_entities, entities_id, relations, heads, sorted_entity_list, Indepth_total_candidates


def question_clearify(question, args, clue = ''):
    system_prompt = prompt_requery_clue + question + "\nClues:"+ clue +'\nOutput:'
    result = run_llm(system_prompt, 0, 512, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), reasoning_effort=getattr(args, "reasoning_effort", None))
    result = extract_answer(result)
    if result:
        return result
    else:
        return question

def reasoning(question, Indepth_total_candidates, Total_Related_Senteces, cluster_chain_of_entities, args, clue):
    def num_tokens_from_string(string: str) -> int:
        """Returns the number of tokens in a text string."""
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    chain_prompt = '('+')\n('.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])+')'

    if ('fever' or 'creak') in args.dataset:
        check_prompt = '### Claim:' + question

        if "fever" in args.dataset:
            if args.clue_query:
                chain_prompt += '\n' + 'clue:' + clue + '\n'
                system_prompt = prompt_reasoning_fever_query_change_3shot_2
            else:
                system_prompt = prompt_reasoning_fever_3shot_2
        else:
            if args.clue_query:
                chain_prompt += '\n' + 'clue:' + clue + '\n'
                system_prompt = prompt_reasoning_creak_query_change_3shot
            else:
                system_prompt = vanilla_prompt_fact_check_3shot


    else:
        check_prompt = '### Question:' + question
        if args.clue_query:
            chain_prompt += '\n' + 'clue:' + clue + '\n'
            system_prompt = prompt_reasoning_qa_query_change_2shot
        else:
            system_prompt = prompt_reasoning_qa_2shot

    sorted_sentences = Total_Related_Senteces[0:args.num_sents_for_reasoning]
    texts=[]
    for sentence in sorted_sentences:
        have_name=False
        for candidate in Indepth_total_candidates:
            if 'related_sentences' in candidate.keys(): 
                for related_sentence in candidate['related_sentences']:
                    if sentence['text'] == related_sentence['text']:
                        have_name=True
                        texts.append('Entity: '+candidate['name']+' Referrence: '+sentence['text'])
        if not have_name:
            texts.append(sentence['text'])

    num_tokens = num_tokens_from_string(system_prompt + "\nKnowledge Triplets:\n" + chain_prompt + '\nRetrieved sentences:\n' + '\nAnswer:')
    related_sentences_prompt = ''
    if len(texts) > 0:
        for i in range(min(args.num_sents_for_reasoning, len(texts))):
            num_tokens += num_tokens_from_string(texts[i]) + 1
            if num_tokens < 4060:
                related_sentences_prompt += '\n' + texts[i]
            else:
                break

    check_prompt += "\n### Knowledge Triplets:\n" + chain_prompt +'\n### Retrieved References:\n' + related_sentences_prompt + '\n### Answer:'

    final_prompt = system_prompt + check_prompt
    response = run_llm(final_prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), reasoning_effort=getattr(args, "reasoning_effort", None))

    print('-----------reasoning result-----------')
    print(response)

    result = extract_answer(response)
    if if_true(result) or contains_yes_regex(response):
        return True, response,final_prompt
    else:
        return False, response,final_prompt


def reasoning_fin(question, Indepth_total_candidates, Total_Related_Senteces, cluster_chain_of_entities, args):


    assert args.dataset == 'finkg_qa'
    check_prompt = '<问题>:\n' + question + "\n"

    sorted_sentences = Total_Related_Senteces[0:args.num_sents_for_reasoning]
    texts = {}

    for candidate in Indepth_total_candidates:
        if 'sentences' in candidate.keys():
            texts[candidate['pre_path']] = []
            for related_sentence in candidate['sentences']:
                for sentence in sorted_sentences:
                    if sentence['text'] == related_sentence['text']:
                        texts[candidate['pre_path']].append(sentence['text'])

    for index, (k, v) in enumerate(texts.items(), start=1):
        check_prompt += "\n<三元组路径{}>:\n".format(index)
        check_prompt += k
        check_prompt += "\n<三元组路径{}中实体{}的相关文本>:\n".format(index, k.split("-")[-1])
        check_prompt += '\n'.join(v)

    check_prompt += "<回答>:\n"


    final_prompt = prompt_reasoning_finqa_query_change_3shot + check_prompt
    response = run_llm_cnfin(final_prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, vendor=getattr(args, "vendor", None), reasoning_effort=getattr(args, "reasoning_effort", None))

    print('-----------reasoning result-----------')
    print(response)

    result = extract_answer(response)
    if if_true(result) or contains_yes_regex(response):
        return True, response, final_prompt
    else:
        return False, response, final_prompt


docs_folder='docs_pages/'

def get_wikipedia_page(wiki_client,entity_dict):

    related_passage = wiki_client.query_all(
        "get_wikipedia_page", entity_dict
    )
    related_passage = "".join(related_passage)

    filename=os.path.join(docs_folder,'{}.txt'.format(entity_dict['name']))
    try:
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(str(related_passage))
    except:
        print(filename+' failed to save！')
    return str(related_passage)


