"""Question cache glue for RoG, reusing ToG's PersistentQuestionCache verbatim.

Same cache, different cached artifact:

  ToG   key=question  value=cluster_chain_of_entities  (KG exploration result)
  RoG   key=question  value=relation paths ("rules")   (planner beam-search result)

In both systems the cached value is everything the pipeline computes *before*
the final answer LLM call, so a hit skips the expensive stage and the answer is
still generated fresh.

One property is specific to RoG and worth stating, because it is what the
experiment is really testing: a RoG relation path is entity-agnostic
(`people.person.sibling_s -> people.sibling_relationship.sibling`). It is
grounded against the *querying* question's own subgraph and its own q_entity in
stage 2. So a semantic hit transplants a reasoning *schema* between questions,
not entities -- unlike ToG, whose cached chain contains concrete entities.
"""

import os
import sys
from pathlib import Path


def _import_question_cache():
    """Import ToG's question_cache module, wherever this is running from.

    Search order: $TOG_CACHE_DIR, /togcache (the docker mount used by
    scripts/run_rog_cache_experiment.sh), then the in-repo path.
    """
    candidates = [
        os.environ.get("TOG_CACHE_DIR"),
        "/togcache",
        str(Path(__file__).resolve().parent.parent / "ToG-cache" / "ToG"),
    ]
    for cand in candidates:
        if cand and (Path(cand) / "question_cache.py").exists():
            sys.path.insert(0, cand)
            import question_cache

            return question_cache
    raise ImportError(
        "could not locate question_cache.py; set TOG_CACHE_DIR to the directory "
        f"holding it (tried: {[c for c in candidates if c]})"
    )


_qc = _import_question_cache()

PersistentQuestionCache = _qc.PersistentQuestionCache
_normalize = _qc._normalize


class TracingQuestionCache(PersistentQuestionCache):
    """PersistentQuestionCache that also reports *why* the last get() hit.

    The base class prints the matched question but does not return it. The
    experiment needs it per-record (which cached question was reused, at what
    cosine similarity) to explain any accuracy delta after the fact.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_hit = None
        self._last_semantic = None

    def _semantic_lookup(self, *args, **kwargs):
        self._last_semantic = super()._semantic_lookup(*args, **kwargs)
        return self._last_semantic

    def get(self, question, oracle_key=None):
        self._last_semantic = None
        exact_before = self.exact_hits
        value = super().get(question, oracle_key=oracle_key)
        if value is None:
            self.last_hit = None
        elif self.exact_hits > exact_before:
            self.last_hit = {"kind": "exact", "source": _normalize(question), "similarity": 1.0}
        elif self._last_semantic is not None:
            matched_key, sim = self._last_semantic
            self.last_hit = {"kind": self.policy, "source": matched_key, "similarity": sim}
        else:  # shouldn't happen; don't lose the hit if it does
            self.last_hit = {"kind": "unknown", "source": None, "similarity": None}
        return value


def extract_oracle_answer_key(sample, dataset):
    """Gold-answer key for the `semantic_oracle` policy.

    Deliberately NOT a reimplementation: this is ToG's own
    question_cache.extract_oracle_answer_key, which grew an `rog-*` branch for
    RoG-format datasets. Both systems call the same function and get the same
    canonical form (stripped, lower-cased gold-answer strings), so an oracle key
    means exactly the same thing in a ToG cache file and a RoG one.
    """
    return _qc.extract_oracle_answer_key(sample, dataset.lower())
