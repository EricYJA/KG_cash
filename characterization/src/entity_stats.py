from collections import Counter
import numpy as np

def compute_redundancy_stats(flat_entities, dataset_name):
    """
    Core characterization: how much redundancy exists in entity mentions?
    """
    counts = Counter(flat_entities)

    total    = len(flat_entities)
    unique   = len(counts)
    repeated = sum(1 for c in counts.values() if c > 1)

    # Perfect cache hit rate:
    # every mention after the first occurrence of an entity is a cache hit
    cache_hits    = sum(c - 1 for c in counts.values())
    hit_rate      = cache_hits / total

    # Top-K coverage
    top10  = sum(c for _, c in counts.most_common(10))
    top50  = sum(c for _, c in counts.most_common(50))
    top100 = sum(c for _, c in counts.most_common(100))

    stats = {
        'dataset'        : dataset_name,
        'total_mentions' : total,
        'unique_entities': unique,
        'repeated_entities': repeated,
        'repeated_pct'   : repeated / unique,
        'cache_hit_rate' : hit_rate,
        'top10_coverage' : top10 / total,
        'top50_coverage' : top50 / total,
        'top100_coverage': top100 / total,
        'counts'         : counts,
    }

    _print_stats(stats)
    return stats

def _print_stats(stats):
    print(f"\n{'='*55}")
    print(f"  Dataset         : {stats['dataset']}")
    print(f"{'='*55}")
    print(f"  Total mentions  : {stats['total_mentions']:,}")
    print(f"  Unique entities : {stats['unique_entities']:,}")
    print(f"  Repeated entities: {stats['repeated_entities']:,} "
          f"({stats['repeated_pct']:.1%} of unique)")
    print(f"  Perfect cache hit rate : {stats['cache_hit_rate']:.2%}")
    print(f"  Top-10  entity coverage: {stats['top10_coverage']:.2%}")
    print(f"  Top-50  entity coverage: {stats['top50_coverage']:.2%}")
    print(f"  Top-100 entity coverage: {stats['top100_coverage']:.2%}")