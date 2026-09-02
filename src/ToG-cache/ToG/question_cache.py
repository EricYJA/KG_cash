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
import random
import threading
import time
from collections import OrderedDict


_VALID_POLICIES = (
    "exact",
    "semantic_lru",
    "semantic_lfu",
    "semantic_fifo",
    "semantic_random",
    "semantic_belady",
    "semantic_oracle",
)
# Belady's MIN needs the whole future request stream, so it is only realisable
# offline. It is accepted here so the simulator can reuse this class, but the
# live _evict_one refuses it rather than silently degrading to FIFO.
_SIMULATION_ONLY_POLICIES = ("semantic_belady",)
# Accept old names as aliases so existing scripts / cache files keep working.
_POLICY_ALIASES = {"semantic": "semantic_lru", "oracle": "semantic_oracle"}
_USES_EMBEDDING = (
    "semantic_lru",
    "semantic_lfu",
    "semantic_fifo",
    "semantic_random",
    "semantic_belady",
    "semantic_oracle",
)
# Policies whose eviction order is insertion order, so a hit must NOT reorder
# the store. Everything else refreshes the entry's position on access (LRU).
_INSERTION_ORDERED = (
    "semantic_lfu",
    "semantic_fifo",
    "semantic_random",
    "semantic_belady",
)
# Semantic policies that match on plain cosine similarity, with no extra
# admission test. `semantic_oracle` is excluded: it also requires a gold-answer
# overlap, so it takes the separate branch in get().
_PLAIN_SEMANTIC = (
    "semantic_lru",
    "semantic_lfu",
    "semantic_fifo",
    "semantic_random",
    "semantic_belady",
)


def _normalize(question: str) -> str:
    return question.strip()


def _cosine_normalized(a, b) -> float:
    # Both vectors are L2-normalized at insertion time, so cosine == dot.
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s


def _select_torch_device() -> str:
    """Pick a device the installed torch build can actually run kernels on.

    `torch.cuda.is_available()` is not enough: a GPU like the GTX 1080 Ti
    (sm_61) shows as available but raises "no kernel image is available
    for execution on the device" when the torch wheel was built only for
    sm_70+. Fall back to CPU in that case.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
    supported_majors: set[int] = set()
    for arch in arch_list:
        if arch.startswith("sm_"):
            try:
                supported_majors.add(int(arch[3:]) // 10)
            except ValueError:
                continue
    for i in range(torch.cuda.device_count()):
        major, _ = torch.cuda.get_device_capability(i)
        if not supported_majors or major in supported_majors:
            return "cuda"
    return "cpu"


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
        self._device = _select_torch_device()
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model_name, device=self._device)
            self._mode = "st"
            return
        except Exception:
            pass
        from transformers import AutoTokenizer, AutoModel
        name = self.model_name if "/" in self.model_name else f"sentence-transformers/{self.model_name}"
        self._tok = AutoTokenizer.from_pretrained(name)
        self._model = AutoModel.from_pretrained(name)
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
        random_seed: int = 0,
    ):
        policy = _POLICY_ALIASES.get(policy, policy)
        if policy not in _VALID_POLICIES:
            raise ValueError(f"policy must be one of {_VALID_POLICIES}, got {policy!r}")
        self.path = path
        self.capacity = capacity
        self.policy = policy
        self.similarity_threshold = similarity_threshold
        self.embedder_model = embedder_model
        self.random_seed = random_seed
        # Own RNG instance so `semantic_random` is reproducible and does not
        # perturb (or get perturbed by) the global random stream, which
        # predict_answer_api.py seeds to pin RoG's path shuffle.
        self._rng = random.Random(random_seed)
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
        self.semantic_fifo_hits = 0
        self.semantic_random_hits = 0
        self.semantic_belady_hits = 0
        self.semantic_oracle_hits = 0
        # Seconds spent inside the cache itself, split by the two things it does.
        # A no-cache run pays neither, so the miss side of this is what has to
        # come back out before a hit can be priced against a miss -- see
        # cache_metrics.baseline_miss_seconds. Cumulative, because a caller
        # brackets a whole question rather than a single call.
        self.lookup_total_s = 0.0
        self.store_total_s = 0.0
        self._embedder: "_Embedder | None" = None
        self._load()

    @property
    def overhead_total_s(self) -> float:
        """Total seconds this cache has spent on lookups and stores.

        Read before and after a question to charge that question its share.
        """
        return self.lookup_total_s + self.store_total_s

    def warm_embedder(self) -> None:
        """Load the sentence embedder now rather than inside the first question.

        The model load costs seconds and lands entirely on whichever question
        happens to be first, where it would be charged as that question's cache
        overhead and subtracted from its (much smaller) traversal time. Paying it
        up front keeps every measured question comparable.
        """
        if self.policy not in _USES_EMBEDDING:
            return
        try:
            self._embed("warmup")
        except Exception as e:
            print(f"[question_cache] embedder warm-up failed, continuing: {e}")

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
        # PID-scoped scratch file: a shared ".tmp" makes two processes writing the
        # same cache path race, and the loser's os.replace dies with ENOENT after
        # the winner consumed the tmp. Each process now renames only its own file.
        # (Concurrent writers still clobber each other's *contents* -- give each
        # run its own --question-cache-path if the entries must stay separate.)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"version": 4, "entries": entries}, f)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _evict_one(self):
        """Evict one entry per the cache's eviction strategy. Caller holds _lock."""
        if not self._store:
            return None
        if self.policy in _SIMULATION_ONLY_POLICIES:
            raise NotImplementedError(
                f"policy {self.policy!r} needs the future request stream and is only "
                f"available under the offline simulator "
                f"(simulate_cache.SemanticBeladyCache), not in a live run"
            )
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
        elif self.policy == "semantic_random":
            evicted = self._rng.choice(list(self._store))
            del self._store[evicted]
        else:
            # LRU (default) and FIFO both pop from the front. They differ only in
            # whether a hit moves the entry to the back -- see _touch/_INSERTION_ORDERED.
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

        Timed: under a semantic policy the lookup embeds the query and scans
        every cached embedding, which is a real per-question cost that a
        no-cache run does not pay.
        """
        started = time.perf_counter()
        try:
            return self._get(question, oracle_key)
        finally:
            self.lookup_total_s += time.perf_counter() - started

    def _get(self, question: str, oracle_key=None):
        key = _normalize(question)
        with self._lock:
            # Kind of the most recent get() ("exact"/"semantic_lru"/... or None on
            # miss), so callers can persist per-question hit type for a restart-safe
            # cache-breakdown reconstruction.
            self.last_hit_kind = None
            if key in self._store:
                self._touch(key)
                self.hits += 1
                self.exact_hits += 1
                self.last_hit_kind = "exact"
                return self._store[key]
            if self.policy in _PLAIN_SEMANTIC:
                sem = self._semantic_lookup(key)
                if sem is not None:
                    matched_key, sim = sem
                    self._touch(matched_key)
                    self.hits += 1
                    label = self.policy
                    if self.policy == "semantic_lfu":
                        self.semantic_lfu_hits += 1
                    elif self.policy == "semantic_fifo":
                        self.semantic_fifo_hits += 1
                    elif self.policy == "semantic_random":
                        self.semantic_random_hits += 1
                    elif self.policy == "semantic_belady":
                        self.semantic_belady_hits += 1
                    else:
                        self.semantic_lru_hits += 1
                    self.last_hit_kind = label
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
                    self.last_hit_kind = "semantic_oracle"
                    print(f"[question_cache] semantic_oracle hit (sim={sim:.3f}) "
                          f"{key[:60]!r} -> {matched_key[:60]!r}")
                    return self._store[matched_key]
            self.misses += 1
            return None

    def _touch(self, key: str) -> None:
        """Bookkeeping on a successful hit: bump freq, refresh LRU position."""
        self._freq[key] = self._freq.get(key, 0) + 1
        if self.policy not in _INSERTION_ORDERED:
            self._store.move_to_end(key)

    def has(self, question: str) -> bool:
        with self._lock:
            return _normalize(question) in self._store

    def put(self, question: str, chain, oracle_key=None) -> None:
        """Store one question's chain and persist the cache.

        Timed for the same reason as get(): embedding the key and rewriting the
        cache file are the cache's own cost, charged to the miss that triggered
        them, and have to be taken back out before that miss can stand in for
        what an uncached run would have spent.
        """
        started = time.perf_counter()
        try:
            self._put(question, chain, oracle_key)
        finally:
            self.store_total_s += time.perf_counter() - started

    def _put(self, question: str, chain, oracle_key=None) -> None:
        key = _normalize(question)
        with self._lock:
            existed = key in self._store
            if existed and self.policy not in _INSERTION_ORDERED:
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

    def restore_many(self, items) -> int:
        """Re-insert entries a previous process stored but this one cannot see.

        `items` is an iterable of (question, chain, oracle_key). Only questions
        the store does not already hold are inserted, in the order given, so a
        cache file that survived is left exactly as it was.

        This is the resume path for the one failure the answers file cannot
        absorb on its own: the results JSONL says a thousand questions are done
        while the cache that was built alongside them is gone (wiped by hand, on
        another mount, lost to a kill between the flush and the next write).
        Without it the resumed run looks up those thousand questions against an
        empty cache and reports a hit rate for a cache that never existed.

        Not routed through put(): put() flushes on every call, which would
        rewrite the whole file once per restored entry, and its timing counters
        would charge the rebuild to the first question of the resumed run.
        """
        restored = 0
        with self._lock:
            for question, chain, oracle_key in items:
                key = _normalize(question)
                if key in self._store:
                    continue
                self._store[key] = chain
                self._freq[key] = self._freq.get(key, 0) + 1
                if self.policy in _USES_EMBEDDING:
                    try:
                        self._embeddings[key] = self._embed(key)
                    except Exception as e:
                        print(f"[question_cache] embed failed while restoring, "
                              f"storing without embedding: {e}")
                if self.policy == "semantic_oracle" and oracle_key:
                    self._oracle_keys[key] = sorted({str(x) for x in oracle_key})
                restored += 1
                while len(self._store) > self.capacity:
                    self._evict_one()
            if restored:
                self._flush()
        return restored

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
                "semantic_fifo_hits": self.semantic_fifo_hits,
                "semantic_random_hits": self.semantic_random_hits,
                "semantic_belady_hits": self.semantic_belady_hits,
                "semantic_oracle_hits": self.semantic_oracle_hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "size": len(self._store),
                "capacity": self.capacity,
                "path": self.path,
                "similarity_threshold": self.similarity_threshold if uses_emb else None,
                "embedder_model": self.embedder_model if uses_emb else None,
            }


def restore_cache_from_answers(cache, answers_path, metrics_path,
                               oracle_key_by_question=None) -> int:
    """Top `cache` back up from a resumed run's own answers file.

    A resumed run skips every question already in the answers JSONL, so any
    chain the cache lost is never recomputed -- the rest of the run then looks
    those questions up against a cache that does not hold them and reports a hit
    rate for a cache that never existed. The answers file already carries what
    is needed to undo that: each record's `reasoning_chains` is the exact object
    the live run passed to put().

    Only questions the metrics sidecar recorded as a *miss* are restored. A hit
    was served a chain belonging to some other question and never entered the
    cache under its own key, so re-inserting it would make the resumed run
    behave differently from an uninterrupted one rather than the same.

    Returns the number of entries restored (0 when the cache survived intact,
    which is the normal case and costs two file reads).
    """
    if cache is None or not answers_path or not os.path.exists(answers_path):
        return 0

    from cache_metrics import read_question_metrics

    chains, order = {}, []
    with open(answers_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = rec.get("question")
            if not isinstance(question, str):
                continue
            key = (question.strip(), rec.get("loop_idx"))
            if key not in chains:
                order.append(key)
            chains[key] = rec.get("reasoning_chains")
    if not order:
        return 0

    metrics = read_question_metrics(metrics_path)
    if not metrics:
        print(f"[question_cache] {len(order)} answered questions in "
              f"{answers_path} but no metrics sidecar to say which of them "
              f"missed; not rebuilding the cache. Hit rates from here on will "
              f"describe a cache that is missing those entries -- pass --fresh "
              f"to start this config over.")
        return 0

    missed = {
        (m["question"].strip(), m.get("loop_idx"))
        for m in metrics
        if isinstance(m.get("question"), str)
        and not m.get("cache_hit") and not m.get("failed")
    }
    lookup = oracle_key_by_question or {}
    items = [(key[0], chains[key], lookup.get(key[0]))
             for key in order if key in missed]
    return cache.restore_many(items)


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
    if dataset.startswith("rog-"):
        # RoG-format datasets (RoG-webqsp, RoG-cwq) carry the gold answer entities
        # in `a_entity` and the answer strings in `answer`. Same canonical form as
        # the branches above: stripped, lower-cased answer strings.
        keys = set()
        for field in ("a_entity", "answer"):
            values = data.get(field) or []
            if isinstance(values, str):
                values = [values]
            for v in values:
                if isinstance(v, dict):
                    v = v.get("answer") or v.get("text") or v.get("name") or v.get("AnswerArgument")
                if v:
                    keys.add(str(v).strip().lower())
        return frozenset(keys) if keys else None
    # Other datasets: caller can extend; oracle policy will degrade to "miss"
    # for any record without an extractable oracle key.
    return None
