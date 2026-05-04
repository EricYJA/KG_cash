"""Persistent per-question LRU cache for ToG cluster_chain_of_entities.

Cache key: the natural-language question (whitespace-stripped).
Cache value: the cluster_chain_of_entities produced by the ToG search loop.

On a hit, the caller can skip every Virtuoso SPARQL call (and every per-loop
LLM scoring call) and feed the cached chain directly into the final
reasoning/answer step.

Cache policies (selected via `policy=`):

- "exact":           only exact-question matches hit. Zero false-positive risk.

- "semantic_lru":    exact match, else embed the query and force a hit on the
                     closest cached question if cosine similarity >=
                     similarity_threshold. LRU eviction.

- "semantic_lfu":    same hit rule as semantic_lru, but evict the
                     least-frequently-used entry on overflow (insertion-order
                     tie-break among entries with equal frequency).

- "semantic_oracle": exact match, else force a hit on the most-cosine-similar
                     cached entry that ALSO shares a gold answer with the
                     query (i.e. cosine >= threshold AND gold-answer-set
                     overlap). Combines semantic retrieval with an
                     accuracy-safety check; serves as the upper bound for
                     accuracy-preserving semantic caching. Requires the caller
                     to pass `oracle_key` (an iterable of canonical
                     gold-answer strings) on both put() and get().
"""

import json
import os
import threading
from collections import OrderedDict


_VALID_POLICIES = ("exact", "semantic_lru", "semantic_lfu", "semantic_oracle")
# Accept old names as aliases so existing scripts / cache files keep working.
_POLICY_ALIASES = {"semantic": "semantic_lru", "oracle": "semantic_oracle"}
_USES_EMBEDDING = ("semantic_lru", "semantic_lfu", "semantic_oracle")


def _normalize(question: str) -> str:
    return question.strip()


def _cosine_normalized(a, b) -> float:
    # Both vectors are L2-normalized at insertion time, so cosine == dot.
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s


class _Embedder:
    """Lazy-loaded sentence embedder.

    Tries sentence-transformers first; falls back to transformers + mean-pool +
    L2-normalize, which reproduces the sentence-transformers recipe.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._mode = None
        self._st = None
        self._tok = None
        self._model = None
        self._device = None

    def _load(self):
        if self._mode is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._st = SentenceTransformer(self.model_name, device=self._device)
            self._mode = "st"
            return
        except Exception:
            pass
        from transformers import AutoTokenizer, AutoModel
        import torch
        name = self.model_name if "/" in self.model_name else f"sentence-transformers/{self.model_name}"
        self._tok = AutoTokenizer.from_pretrained(name)
        self._model = AutoModel.from_pretrained(name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device).eval()
        self._mode = "hf"

    def encode(self, text: str):
        self._load()
        if self._mode == "st":
            v = self._st.encode([text], normalize_embeddings=True)[0]
            return [float(x) for x in v]
        import torch
        with torch.no_grad():
            enc = self._tok(text, padding=True, truncation=True, return_tensors="pt").to(self._device)
            out = self._model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(summed / counts, p=2, dim=1)[0].cpu().tolist()
        return v


class PersistentQuestionCache:
    def __init__(
        self,
        path: str,
        capacity: int = 4096,
        policy: str = "exact",
        similarity_threshold: float = 0.95,
        embedder_model: str = "all-MiniLM-L6-v2",
    ):
        policy = _POLICY_ALIASES.get(policy, policy)
        if policy not in _VALID_POLICIES:
            raise ValueError(f"policy must be one of {_VALID_POLICIES}, got {policy!r}")
        self.path = path
        self.capacity = capacity
        self.policy = policy
        self.similarity_threshold = similarity_threshold
        self.embedder_model = embedder_model
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, list]" = OrderedDict()
        self._embeddings: "dict[str, list]" = {}
        self._oracle_keys: "dict[str, list[str]]" = {}
        self._freq: "dict[str, int]" = {}
        self.hits = 0
        self.misses = 0
        self.exact_hits = 0
        self.semantic_lru_hits = 0
        self.semantic_lfu_hits = 0
        self.semantic_oracle_hits = 0
        self._embedder: "_Embedder | None" = None
        self._load()

    def _embed(self, q: str):
        if self._embedder is None:
            self._embedder = _Embedder(self.embedder_model)
        return self._embedder.encode(q)

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._store = OrderedDict()
            self._embeddings = {}
            self._oracle_keys = {}
            return
        if not isinstance(payload, dict):
            return
        for item in payload.get("entries", []):
            # Legacy v1: [question, chain]
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
                self._store[item[0]] = item[1]
                continue
            # v2 / v3 / v4: dict with question, chain, optional emb / oracle_key / freq
            if isinstance(item, dict) and isinstance(item.get("question"), str):
                k = item["question"]
                self._store[k] = item.get("chain")
                emb = item.get("emb")
                if isinstance(emb, list) and emb:
                    self._embeddings[k] = emb
                ok = item.get("oracle_key")
                if isinstance(ok, list) and ok:
                    self._oracle_keys[k] = [str(x) for x in ok]
                fr = item.get("freq")
                if isinstance(fr, int) and fr > 0:
                    self._freq[k] = fr
        while len(self._store) > self.capacity:
            self._evict_one()

    def _flush(self) -> None:
        if not self.path:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        entries = []
        for k, v in self._store.items():
            entry = {"question": k, "chain": v}
            emb = self._embeddings.get(k)
            if emb is not None:
                entry["emb"] = emb
            ok = self._oracle_keys.get(k)
            if ok is not None:
                entry["oracle_key"] = ok
            fr = self._freq.get(k)
            if fr is not None:
                entry["freq"] = fr
            entries.append(entry)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 4, "entries": entries}, f)
        os.replace(tmp, self.path)

    def _evict_one(self):
        """Evict one entry per the cache's eviction strategy. Caller holds _lock."""
        if not self._store:
            return None
        if self.policy == "semantic_lfu":
            # Find min frequency; tie-break by insertion order (first hit in iteration).
            min_freq = None
            evicted = None
            for k in self._store:
                f = self._freq.get(k, 0)
                if min_freq is None or f < min_freq:
                    min_freq = f
                    evicted = k
            del self._store[evicted]
        else:
            # LRU (default): pop the least-recently-touched entry.
            evicted, _ = self._store.popitem(last=False)
        self._embeddings.pop(evicted, None)
        self._oracle_keys.pop(evicted, None)
        self._freq.pop(evicted, None)
        return evicted

    def _semantic_lookup(self, query_key: str, query_oracle_key=None,
                         require_oracle: bool = False):
        """Find the most-cosine-similar cached entry above the threshold.

        If `require_oracle` is True, the candidate must also share a gold
        answer with `query_oracle_key`.
        Caller holds _lock. Returns (key, sim) or None.
        """
        if not self._embeddings:
            return None
        if require_oracle:
            if not query_oracle_key:
                return None
            qset = {str(x) for x in query_oracle_key}
            if not qset:
                return None
        try:
            qv = self._embed(query_key)
        except Exception as e:
            print(f"[question_cache] embed failed, skipping semantic lookup: {e}")
            return None
        best_key, best_sim = None, -1.0
        for k, ev in self._embeddings.items():
            s = _cosine_normalized(qv, ev)
            if s < self.similarity_threshold:
                continue
            if require_oracle:
                ok = self._oracle_keys.get(k)
                if not ok or not qset.intersection(ok):
                    continue
            if s > best_sim:
                best_sim = s
                best_key = k
        if best_key is None:
            return None
        return best_key, best_sim

    def get(self, question: str, oracle_key=None):
        """Return cached chain, or None on miss.

        An empty list IS a valid cached value (ToG previously found no chain).
        Use `has` to disambiguate miss-vs-empty.

        `oracle_key`: only consulted under policy="semantic_oracle". Iterable
        of canonical gold-answer strings for the query.
        """
        key = _normalize(question)
        with self._lock:
            if key in self._store:
                self._touch(key)
                self.hits += 1
                self.exact_hits += 1
                return self._store[key]
            if self.policy in ("semantic_lru", "semantic_lfu"):
                sem = self._semantic_lookup(key)
                if sem is not None:
                    matched_key, sim = sem
                    self._touch(matched_key)
                    self.hits += 1
                    if self.policy == "semantic_lfu":
                        self.semantic_lfu_hits += 1
                        label = "semantic_lfu"
                    else:
                        self.semantic_lru_hits += 1
                        label = "semantic_lru"
                    print(f"[question_cache] {label} hit (sim={sim:.3f}) "
                          f"{key[:60]!r} -> {matched_key[:60]!r}")
                    return self._store[matched_key]
            elif self.policy == "semantic_oracle":
                sem = self._semantic_lookup(key, oracle_key, require_oracle=True)
                if sem is not None:
                    matched_key, sim = sem
                    self._touch(matched_key)
                    self.hits += 1
                    self.semantic_oracle_hits += 1
                    print(f"[question_cache] semantic_oracle hit (sim={sim:.3f}) "
                          f"{key[:60]!r} -> {matched_key[:60]!r}")
                    return self._store[matched_key]
            self.misses += 1
            return None

    def _touch(self, key: str) -> None:
        """Bookkeeping on a successful hit: bump freq, refresh LRU position."""
        self._freq[key] = self._freq.get(key, 0) + 1
        if self.policy != "semantic_lfu":
            self._store.move_to_end(key)

    def has(self, question: str) -> bool:
        with self._lock:
            return _normalize(question) in self._store

    def put(self, question: str, chain, oracle_key=None) -> None:
        key = _normalize(question)
        with self._lock:
            existed = key in self._store
            if existed and self.policy != "semantic_lfu":
                self._store.move_to_end(key)
            self._store[key] = chain
            # Treat put as an access: new entries start at freq=1, repeats bump.
            self._freq[key] = self._freq.get(key, 0) + 1
            if self.policy in _USES_EMBEDDING:
                try:
                    self._embeddings[key] = self._embed(key)
                except Exception as e:
                    print(f"[question_cache] embed failed on put, storing without embedding: {e}")
            if self.policy == "semantic_oracle" and oracle_key:
                self._oracle_keys[key] = sorted({str(x) for x in oracle_key})
            while len(self._store) > self.capacity:
                self._evict_one()
            self._flush()

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            uses_emb = self.policy in _USES_EMBEDDING
            return {
                "policy": self.policy,
                "hits": self.hits,
                "exact_hits": self.exact_hits,
                "semantic_lru_hits": self.semantic_lru_hits,
                "semantic_lfu_hits": self.semantic_lfu_hits,
                "semantic_oracle_hits": self.semantic_oracle_hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "size": len(self._store),
                "capacity": self.capacity,
                "path": self.path,
                "similarity_threshold": self.similarity_threshold if uses_emb else None,
                "embedder_model": self.embedder_model if uses_emb else None,
            }


def extract_oracle_answer_key(data: dict, dataset: str):
    """Return a frozenset of canonical gold-answer strings for `data`, or None.

    Two questions whose oracle keys share any element are considered
    oracle-equivalent: a chain cached for one is reusable for the other,
    because it should still lead to a correct answer.
    """
    if dataset == "webqsp":
        keys = set()
        for parse in data.get("Parses", []) or []:
            for ans in parse.get("Answers", []) or []:
                v = ans.get("EntityName") or ans.get("AnswerArgument")
                if v:
                    keys.add(str(v).strip().lower())
        return frozenset(keys) if keys else None
    if dataset == "cwq":
        keys = set()
        ans_field = data.get("answer")
        if isinstance(ans_field, list):
            for a in ans_field:
                if isinstance(a, dict):
                    v = a.get("answer") or a.get("text") or a.get("name") or a.get("AnswerArgument")
                else:
                    v = a
                if v:
                    keys.add(str(v).strip().lower())
        elif isinstance(ans_field, str):
            keys.add(ans_field.strip().lower())
        return frozenset(keys) if keys else None
    # Other datasets: caller can extend; oracle policy will degrade to "miss"
    # for any record without an extractable oracle key.
    return None
