import re
import os
import json
try:
    from blingfire import text_to_sentences_and_offsets
except ImportError:
    def text_to_sentences_and_offsets(text):
        offsets = []
        start = 0
        for match in re.finditer(r"[^.!?]+[.!?]?", text):
            sentence = match.group(0).strip()
            if sentence:
                offsets.append((match.start(), match.end()))
        return text, offsets

docs_folder='docs_pages/'

if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)

def scores_rank(scores,texts):
    items=[]
    for i in range(len(scores)):
        item={}
        item['score']=float(scores[i])
        item['text']=texts[i]
        items.append(item)

    sorted_data = sorted(items, key=lambda x: x['score'], reverse=True)

    return sorted_data

def crossencoder_similarity(question, texts, args, emb_model):
    assert args.embedding_model_name == 'minilm' or 'bge-ce'
    if len(texts) == 1 :
        l = [[question, texts[0] +' ']]
    elif len(texts) == 0:
        return []
    else:
        l = [[question, text] for text in texts]
    if args.embedding_model_name == 'bge-ce':
        scores = emb_model.compute_score(l)
    else:
        scores = emb_model.predict(l)


    return scores

def biencoder_similarity(question, passages, args, emb_model):
    assert args.embedding_model_name == 'bge-bi'
    questions = [question] * len(passages)
    q_embeddings = emb_model.encode_queries(questions)
    p_embeddings = emb_model.encode(passages)
    scores = q_embeddings @ p_embeddings.T
    return scores[0]

def s2p_relevance_scores(texts, question, args, emb_model):
    # print(args.embedding_model_name)
    if args.embedding_model_name == 'bge-bi':
        scores = biencoder_similarity(question, texts, args, emb_model)
        return scores
    elif args.embedding_model_name == 'minilm' or args.embedding_model_name == 'bge-ce':
        scores = crossencoder_similarity(question, texts, args, emb_model)
        return scores
    elif args.embedding_model_name == 'bm25':
        scores = emb_model(question, texts)
        return scores
    elif args.embedding_model_name == 'colbert':
        q_embeddings = emb_model.encode([question], return_dense=False, return_sparse=False, return_colbert_vecs=True)
        p_embeddings = emb_model.encode(texts, return_dense=False, return_sparse=False, return_colbert_vecs=True)
        scores = []
        for i in range(len(texts)):
            scores.append(emb_model.colbert_score(q_embeddings['colbert_vecs'][0], p_embeddings['colbert_vecs'][i]))
        return scores
    else:
        raise Exception('Unknown embedding model')


def split_paragraphs(text):
    paragraphs = []
    current_paragraph = ''
    for line in text.splitlines():
        if line.strip() == '':
            if current_paragraph != '':
                paragraphs.append(current_paragraph)
                current_paragraph = ''
        else:
            current_paragraph += line + '\n'

    paragraphs_final=[]
    for p in paragraphs:
        p=re.sub(r'\[\d+\]', '', p)
        p=p.rstrip('\n')
        paragraphs_final.append(p)

    filtered_paragraphs=[]
    for p in paragraphs_final:
        if not p.strip().startswith('^') :
            if len(p.strip().split('\n')) > 1:
                if len(p)>=50 :
                    rows = p.split('\n')
                    if len(p)/len(rows) >= 50:
                        filtered_paragraphs.append(p)
    return filtered_paragraphs

def split_sentences_1(text):
    sentences=re.split('\.', text)
    filtered_sentences = [sentence for sentence in sentences if len(sentence) >= 20]
    filtered_sentences = [sentence.strip() for sentence in filtered_sentences if sentence.strip()]
    return filtered_sentences


def split_sentences_windows(text, window_size=2, step_size=1):
    all_sentences = []
    if len(text) > 0:
        offsets = text_to_sentences_and_offsets(text)[1]
        for ofs in offsets:
            sentence = text[ofs[0]: ofs[1]]
            all_sentences.append(
                sentence
            )
    else:
        all_sentences.append("")

    if window_size > 1 and len(all_sentences) >= (window_size + step_size):
        all_windows = []

        for i in range(0, len(all_sentences) - window_size + 1, step_size):
            window = all_sentences[i:i + window_size]
            combined_sentence = ' '.join(window)
            all_windows.append(combined_sentence)
        all_sentences = all_windows
    return all_sentences

def split_sentences(text):

    paragraphs = text.split('\n')

    filtered_paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) >= 10]
    all_text = '.'.join(filtered_paragraphs)

    sentences = re.split('\.', all_text)
    filtered_sentences = [sentence for sentence in sentences if len(sentence) >= 10]

    filtered_sentences = [sentence.strip() for sentence in filtered_sentences if sentence.strip()]

    return filtered_sentences

def pages_embedding_search(question,related_passage, args, emb_model,top_k=3):
    content=related_passage

    if content!='Not Found!':

        splited_paragraphs = split_paragraphs(content)

        if len(splited_paragraphs) < 1:
            return '',[]
        scores = s2p_relevance_scores(splited_paragraphs, question, args, emb_model)
        sorted_splited_paragraphs = scores_rank(scores, splited_paragraphs)

        if len(sorted_splited_paragraphs) >= 3:
            paragraph = sorted_splited_paragraphs[0]['text'] + sorted_splited_paragraphs[1]['text'] + sorted_splited_paragraphs[2]['text']
        else:
            paragraph = ''.join([p['text'] for p in sorted_splited_paragraphs])

        splited_sentences=split_sentences_windows(paragraph)

        scores = s2p_relevance_scores(splited_sentences, question, args, emb_model)
        sorted_sentences = scores_rank(scores, splited_sentences)

        return paragraph,sorted_sentences[0:top_k]
    else:
        return '',[]

def pages_embedding_search_only_para(related_passage):
    content=related_passage

    if content!='Not Found!':

        splited_paragraphs = split_paragraphs(content)

        return splited_paragraphs
    else:
        return []

def save_2_jsonl_simplier(question,ground_truth, answer, search_entity_list,Total_Related_Senteces, cluster_chain_of_entities, dataset,end_mode,remark,args):
    data_dict = {"question": question,
                 'ground_truth': ground_truth,
                 'answer':answer,
                 "end_mode":end_mode,
                 "remark":remark,
                 }

    tp = '_TP' if args.topic_prune else ''
    rc = '_RC' if args.relation_prune_combination else ''
    filename = "shot-{}-{}-{}-w{}d{}s{}sw{}_{}".format(
        args.LLM_type,
        args.dataset,
        args.embedding_model_name,
        args.width,
        args.depth,
        args.num_sents_for_reasoning,
        args.sliding_window[0],
        args.sliding_window[1]
    )
    if args.self_consistency:
        filename += '_slf{}'.format(args.self_consistency_threshold)
    filename += tp + rc + '.json'
    if args.gpt_only:
        filename = '{}_{}.json'.format(args.LLM_type,args.dataset)

    filename = getattr(args, "output_file", None) or filename
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(filename):

        with open(filename, "r") as f:
            content = f.read()
            if content.endswith("]"):

                content = content.strip().rstrip("]")

                content += "," + json.dumps(data_dict, ensure_ascii=False, indent=4) + "]"
            else:
                content = "["+json.dumps(data_dict, ensure_ascii=False, indent=4) + "]"
    else:
        content = "["+json.dumps(data_dict, ensure_ascii=False, indent=4) + "]"

    with open(filename, "w") as f:
        f.write(content)



