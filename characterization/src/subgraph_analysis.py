import numpy as np
from collections import defaultdict

def analyze_subgraph_overlap(dataset, split='train'):
    """
    For each topic entity, collect all triples ever associated with it.
    This tells you: if we cached by entity, how large would each cached
    subgraph be, and how much triple-level reuse exists?
    """
    entity_to_triples = defaultdict(set)
    entity_to_questions = defaultdict(list)

    for i, sample in enumerate(dataset[split]):
        entities = sample['q_entity']
        if isinstance(entities, str):
            entities = [entities]

        triples = set(tuple(t) for t in sample['graph'])

        for entity in entities:
            entity_to_triples[entity].update(triples)
            entity_to_questions[entity].append(i)

    # Subgraph size stats
    sizes = [len(triples) for triples in entity_to_triples.values()]
    sizes = np.array(sizes)

    # Triple-level reuse: how many triples appear in >1 question?
    triple_counts = {}
    for sample in dataset[split]:
        for triple in sample['graph']:
            key = tuple(triple)
            triple_counts[key] = triple_counts.get(key, 0) + 1

    reused_triples = sum(1 for c in triple_counts.values() if c > 1)
    total_unique_triples = len(triple_counts)

    stats = {
        'entity_to_triples'   : entity_to_triples,
        'entity_to_questions' : entity_to_questions,
        'subgraph_sizes'      : sizes,
        'total_unique_triples': total_unique_triples,
        'reused_triples'      : reused_triples,
        'triple_reuse_rate'   : reused_triples / total_unique_triples,
    }

    _print_subgraph_stats(stats)
    return stats

def _print_subgraph_stats(stats):
    sizes = stats['subgraph_sizes']
    print(f"\n  Subgraph size (triples per entity):")
    print(f"    Mean   : {np.mean(sizes):.1f}")
    print(f"    Median : {np.median(sizes):.1f}")
    print(f"    Max    : {np.max(sizes)}")
    print(f"    Min    : {np.min(sizes)}")
    print(f"  Total unique triples  : {stats['total_unique_triples']:,}")
    print(f"  Reused triples (>1q)  : {stats['reused_triples']:,}")
    print(f"  Triple reuse rate     : {stats['triple_reuse_rate']:.2%}")