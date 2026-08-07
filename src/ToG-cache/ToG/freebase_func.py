import os

from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed
# Any SPARQL 1.1 endpoint serving the Freebase triples works here (Virtuoso,
# Oxigraph, ...); see docker-compose.yml. Default is the local Virtuoso.
SPARQLPATH = os.environ.get("SPARQL_ENDPOINT", "http://localhost:8890/sparql")

# pre-defined sparqls
#
# Absolute IRIs on purpose: prefixed names like ns:type.object.name (two dots
# in the local part) are valid SPARQL but rejected by some strict parsers
# (e.g. Oxigraph), and mids/relations are substituted in as raw strings. Full
# IRIs parse identically on every backend.
sparql_head_relations = """\nSELECT ?relation\nWHERE {\n  <http://rdf.freebase.com/ns/%s> ?relation ?x .\n}"""
sparql_tail_relations = """\nSELECT ?relation\nWHERE {\n  ?x ?relation <http://rdf.freebase.com/ns/%s> .\n}"""
sparql_tail_entities_extract = """SELECT ?tailEntity\nWHERE {\n<http://rdf.freebase.com/ns/%s> <http://rdf.freebase.com/ns/%s> ?tailEntity .\n}"""
sparql_head_entities_extract = """SELECT ?tailEntity\nWHERE {\n?tailEntity <http://rdf.freebase.com/ns/%s> <http://rdf.freebase.com/ns/%s>  .\n}"""
sparql_id = """SELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity <http://rdf.freebase.com/ns/type.object.name> ?tailEntity .\n    FILTER(?entity = <http://rdf.freebase.com/ns/%s>)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = <http://rdf.freebase.com/ns/%s>)\n  }\n}"""
    
def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


def execurte_sparql(sparql_txt):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except QueryBadFormed:
        print(f"[sparql] skipping malformed query: {sparql_txt!r}")
        return []
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


def id2entity_name_or_type(entity_id):
    sparql_txt = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']
