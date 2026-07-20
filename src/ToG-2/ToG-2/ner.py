
import os, sys, json
AZURE_CLIENT = None
def get_azure_client():
    global AZURE_CLIENT
    if AZURE_CLIENT is not None:
        return AZURE_CLIENT

    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    key = '<your_azure_key>'
    endpoint = '<your_azure_url>'

    ta_credential = AzureKeyCredential(key)
    AZURE_CLIENT = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return AZURE_CLIENT


def entity_linking_azure_search(question: str):
    import requests
    from bs4 import BeautifulSoup

    client = get_azure_client()
    try:
        result = client.recognize_linked_entities(documents=[question])[0]
    except Exception as err:
        print("Encountered exception. {}".format(err))
        raise err

    nodes = {}
    for entity in result.entities:
        name = entity.name.replace(' ', '%20')
        qid_url = f'https://en.wikipedia.org/w/index.php?title={name}&action=info'
        try:
            response = requests.get(qid_url)
            page = BeautifulSoup(response.text, "html.parser")
            table = page.findChildren('table', class_='wikitable mw-page-info')[0]
            rows = [i.findChildren('td') for i in table.findChildren('tr')]
            rows = [i for i in rows if i[0].string == 'Wikidata item ID']
        except Exception as err:
            print(qid_url, 'fails')
            raise err

        if not rows:
            continue

        row = rows[0]
        qid = str(row[1].string)
        nodes[qid] = entity.name
    return nodes


_entity_azure_cache = None
def entity_linking_azure_cache(question: str, args):
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    global _entity_azure_cache
    fpath = args.entity_cache if args.entity_cache else os.path.join(ROOT_DIR, f'data/{args.dataset}_entities_azure.json')
    if _entity_azure_cache is None:
        ent_list = json.load(open(fpath))
        ent_dict = {i['question']: i['entities'] for i in ent_list}
        _entity_azure_cache = ent_dict
    ent_dict = _entity_azure_cache  # {qid: name}

    if question not in ent_dict:
        raise ValueError(f'question={question} fails in entity_linking_azure_cache, fpath={fpath}')

    nodes = ent_dict[question]
    return nodes


