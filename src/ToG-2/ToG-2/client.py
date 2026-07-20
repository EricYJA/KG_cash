import itertools
import xmlrpc.client
import typing as tp

def format_entity_name_for_wikipedia(entity_name):
    return entity_name.replace(' ', '_')


class WikidataQueryClient:
    def __init__(self, url: str):
        self.url = url
        self.server = xmlrpc.client.ServerProxy(url)

    def label2qid(self, label: str) -> str:
        return self.server.label2qid(label)

    def label2pid(self, label: str) -> str:
        return self.server.label2pid(label)

    def pid2label(self, pid: str) -> str:
        return self.server.pid2label(pid)

    def qid2label(self, qid: str) -> str:
        return self.server.qid2label(qid)

    def get_all_relations_of_an_entity(
        self, entity_qid: str
    ) -> tp.Dict[str, tp.List]:
        return self.server.get_all_relations_of_an_entity(entity_qid)

    def get_tail_entities_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.Dict[str, tp.List]:
        return self.server.get_tail_entities_given_head_and_relation(
            head_qid, relation_pid
        )

    def get_tail_values_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.List[str]:
        return self.server.get_tail_values_given_head_and_relation(
            head_qid, relation_pid
        )

    def get_external_id_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.List[str]:
        return self.server.get_external_id_given_head_and_relation(
            head_qid, relation_pid
        )

    def get_wikipedia_page(self, ent_dict, section: str = None) -> str:
        try:
            if ent_dict.get('name') and ent_dict['name'] != "Not Found!":
                entity_name = format_entity_name_for_wikipedia(ent_dict['name'])
            elif ent_dict['id'] != 'None':
                qid = ent_dict['id']
                entity_name = self.server.get_wikipedia_link(qid)
                entity_name = entity_name[0]
            else:
                return "Not Found!"

            if entity_name == "Not Found!":
                return "Not Found!"
            else:
                import requests
                from bs4 import BeautifulSoup

                wikipedia_url = 'https://en.wikipedia.org/wiki/{}'.format(entity_name)
                print('wikipedia_url  ' + wikipedia_url)

                response = requests.get(wikipedia_url, headers={'Connection': 'close'}, timeout=180)
                response.raise_for_status()  

                soup = BeautifulSoup(response.content, "html.parser")
                content_div = soup.find("div", {"id": "bodyContent"})

                # Remove script and style elements
                for script_or_style in content_div.find_all(["script", "style"]):
                    script_or_style.decompose()

                if section:
                    header = content_div.find(
                        lambda tag: tag.name == "h2" and section in tag.get_text()
                    )
                    if header:
                        content = ""
                        for sibling in header.find_next_siblings():
                            if sibling.name == "h2":
                                break
                            content += sibling.get_text()
                        return content.strip()
                    else:
                        return f"Section '{section}' not found."

                summary_content = ""
                for element in content_div.find_all(recursive=False):
                    if element.name == "h2":
                        break
                    summary_content += element.get_text()

                return summary_content.strip()
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Wikipedia page: {e}")
            return "Not Found!"

    def mid2qid(self, mid: str) -> str:
        return self.server.mid2qid(mid)


import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor


class MultiServerWikidataQueryClient:
    def __init__(self, urls: tp.List[str]):
        self.clients = [WikidataQueryClient(url) for url in urls]
        self.executor = ThreadPoolExecutor(max_workers=len(urls))
       

    def test_connections(self):
        def test_url(client):
            try:
               
                client.server.system.listMethods()
                return True
            except Exception as e:
                print(f"Failed to connect to {client.url}. Error: {str(e)}")
                return False

        start_time = time.perf_counter()
        futures = [
            self.executor.submit(test_url, client) for client in self.clients
        ]
        results = [f.result() for f in futures]
        end_time = time.perf_counter()
        print(f"Testing connections took {end_time - start_time} seconds")
       
        self.clients = [
            client for client, result in zip(self.clients, results) if result
        ]
        if not self.clients:
            raise Exception("Failed to connect to all URLs")

   
    
    def query_all(self, method, *args):
        # start_time = time.perf_counter()
        futures = [
            self.executor.submit(getattr(client, method), *args) for client in self.clients
        ]
      
        is_dict_return = method in [
            "get_all_relations_of_an_entity",
            "get_tail_entities_given_head_and_relation",
        ]
        results = [f.result() for f in futures]
        # end_time = time.perf_counter()
      

        # start_time = time.perf_counter()
        real_results = (
            set() if not is_dict_return else {"head": [], "tail": []}
        )
        for res in results:
            if isinstance(res, str) and res == "Not Found!":
                continue
            elif isinstance(res, tp.List):
                if len(res) == 0:
                    continue
                if isinstance(res[0], tp.List):
                    res_flattened = itertools.chain(*res)
                    real_results.update(res_flattened)
                    continue
                real_results.update(res)
            elif is_dict_return:
                real_results["head"].extend(res["head"])
                real_results["tail"].extend(res["tail"])
            else:
                real_results.add(res)
        # end_time = time.perf_counter()

        return real_results if len(real_results) > 0 else "Not Found!"


if __name__ == "__main__":
    

    with open("server_urls.txt", "r") as f:
        server_addrs = f.readlines()
        server_addrs = [addr.strip() for addr in server_addrs]
    print(f"Server addresses: {server_addrs}")



    wiki_client = MultiServerWikidataQueryClient(server_addrs)

    entity_candidates_id=['Q47887']
    entity_candidates_id=['Q142']

    # method 1
    # related_passage = wiki_client.clients[0].get_wikipedia_page(entity_candidates_id[0])
    # print('related_passage')
    # print(related_passage)
    # print(type(related_passage)) # str

    # method 2
        
    related_passage = wiki_client.query_all(
        "get_wikipedia_page", entity_candidates_id[0]
    )
    #print(related_passage)
    print(type(related_passage)) # set
    print(len(related_passage))
    related_passage = "".join(related_passage)
    



