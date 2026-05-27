"""WeaviateStore: unified BM25 + vector store backed by Weaviate (v4 client).

This module mirrors the public surface of FAISSStore and BM25Store so a future
integration phase can swap the storage layer without touching retrieval code.

Design notes
------------
- weaviate-client v4 (collections API) is required (>=4.9, <5).
- The Weaviate client is lazily constructed on first use so importing this
  module does not require the SDK to be reachable or even installed at import
  time. Thread-safe construction is guarded by a ``threading.Lock``.
- The schema (``DocChunk`` collection) is auto-provisioned on first connect via
  :func:`ensure_collection`, which is idempotent.
- Vectors are produced by an externally supplied ``embedding_fn`` (sync or
  async) -- the Weaviate vectorizer is disabled.
- ``add_chunks`` accepts either the existing :class:`Chunk` dataclass or an
  enriched dict carrying the full Phase-2 metadata (parent_id, header_path,
  campus, doc_type, faculty, major, year, chunk_level, is_table, section,
  content/text).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from dataclasses import is_dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union
from urllib.parse import urlparse

from src.config.settings import settings

log = logging.getLogger(__name__)

# Schema/collection name is read from settings each call so test overrides work.
_DEFAULT_CONTENT_FIELDS: tuple[str, ...] = ("content", "text")

# Properties known to the DocChunk schema. Anything else on an incoming dict
# is silently dropped to avoid Weaviate "unknown property" insert errors.
_SCHEMA_PROPS: frozenset[str] = frozenset(
    {
        "chunk_id",
        "parent_id",
        "file_id",
        "key",
        "content",
        "section",
        "header_path",
        "chunk_level",
        "is_table",
        "campus",
        "doc_type",
        "faculty",
        "major",
        "year",
    }
)


def _recover_utf8(text: str) -> str:
    """Recover Vietnamese diacritics from cp1252+surrogateescape mojibake.

    Workaround for a long-lived-worker corruption observed with
    weaviate-client 4.18 + Weaviate 1.27 where multi-byte UTF-8 sequences
    get stored as Latin-1 codepoints with bytes 0x80-0x9F escaped as
    surrogates (``Đ`` → ``Ä\\udc90``). The pattern round-trips losslessly
    via ``encode('latin-1', surrogateescape) → decode('utf-8')``.
    """
    if not isinstance(text, str) or not text:
        return text
    # Heuristic: only attempt recovery when the string contains surrogates
    # (the unambiguous fingerprint) OR Latin-1 codepoints in the 0x80-0xFF
    # range alongside non-ASCII text.
    has_surrogate = any(0xDC80 <= ord(c) <= 0xDCFF for c in text)
    has_latin1_hi = any(0x80 <= ord(c) <= 0xFF for c in text)
    if not has_surrogate and not has_latin1_hi:
        return text
    try:
        return text.encode("latin-1", errors="surrogateescape").decode(
            "utf-8", errors="replace"
        )
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class WeaviateStore:
    """Unified hybrid (BM25 + vector) store backed by Weaviate.

    The store exposes the union of FAISSStore + BM25Store public methods:
    ``add_chunks``, ``delete_by_key``, ``search``, ``has_key``,
    ``precompute_embeddings``.

    Parameters
    ----------
    embedding_fn : Callable[[List[str]], Any]
        Callable that maps a list of texts to a list/array of float vectors.
        May be either synchronous or a coroutine function.
    embedding_dim : int
        Dimensionality of the embeddings. Stored for documentation; the schema
        does not pin a vector size in v4 (Weaviate infers from the first insert).
    url : str | None
        Full Weaviate HTTP URL (e.g. ``http://localhost:8080``). Defaults to
        ``settings.WEAVIATE_URL``.
    grpc_host : str | None
        gRPC host name. Defaults to ``settings.WEAVIATE_GRPC_HOST``.
    grpc_port : int | None
        gRPC port. Defaults to ``settings.WEAVIATE_GRPC_PORT``.
    api_key : str | None
        Optional API key. If empty/None, anonymous connection is used and we
        call ``weaviate.connect_to_local``. Otherwise we call
        ``weaviate.connect_to_custom`` and pass an ``Auth.api_key`` credential.
    class_name : str | None
        Override the collection (class) name. Defaults to
        ``settings.WEAVIATE_CLASS_NAME``.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        embedding_fn: Callable[[List[str]], Any],
        embedding_dim: int,
        *,
        url: str | None = None,
        grpc_host: str | None = None,
        grpc_port: int | None = None,
        api_key: str | None = None,
        class_name: str | None = None,
    ) -> None:
        self._embed = embedding_fn
        self._dim = int(embedding_dim)
        self._url = url if url is not None else settings.WEAVIATE_URL
        self._grpc_host = grpc_host if grpc_host is not None else settings.WEAVIATE_GRPC_HOST
        self._grpc_port = int(grpc_port if grpc_port is not None else settings.WEAVIATE_GRPC_PORT)
        self._api_key = api_key if api_key is not None else settings.WEAVIATE_API_KEY
        self._class_name = class_name or settings.WEAVIATE_CLASS_NAME

        self._client: Any | None = None  # weaviate.WeaviateClient, typed lazily
        self._client_lock = threading.Lock()
        # Embedding cache, mirrors FAISSStore semantics: filled by
        # precompute_embeddings, drained on next add_chunks.
        self._embed_cache: Dict[str, List[float]] = {}

    # ------------------------------------------------------------- connect
    def _connect(self) -> Any:
        """Lazily build and cache the v4 Weaviate client.

        Thread-safe via a double-checked lock so concurrent ingestion workers
        do not race on the first call.
        """
        if self._client is not None and getattr(self._client, "is_connected", lambda: True)():
            return self._client
        with self._client_lock:
            if self._client is not None and getattr(self._client, "is_connected", lambda: True)():
                return self._client
            # Import lazily so that the rest of the pipeline (eval scripts,
            # router service, ...) does not have to load weaviate-client.
            import weaviate
            from weaviate.classes.init import AdditionalConfig, Timeout
            from weaviate.auth import Auth

            parsed = urlparse(self._url)
            http_host = parsed.hostname or "localhost"
            http_port = parsed.port or (443 if parsed.scheme == "https" else 8080)
            http_secure = parsed.scheme == "https"

            timeout = Timeout(
                init=int(settings.WEAVIATE_REQUEST_TIMEOUT),
                query=int(settings.WEAVIATE_REQUEST_TIMEOUT),
                insert=int(settings.WEAVIATE_REQUEST_TIMEOUT * 6),
            )
            additional = AdditionalConfig(timeout=timeout)

            t0 = time.perf_counter()
            if self._api_key:
                log.info(
                    "WeaviateStore: connecting to %s (gRPC %s:%s) with API key",
                    self._url, self._grpc_host, self._grpc_port,
                )
                client = weaviate.connect_to_custom(
                    http_host=http_host,
                    http_port=http_port,
                    http_secure=http_secure,
                    grpc_host=self._grpc_host,
                    grpc_port=self._grpc_port,
                    grpc_secure=http_secure,
                    auth_credentials=Auth.api_key(self._api_key),
                    additional_config=additional,
                )
            else:
                log.info(
                    "WeaviateStore: connecting to %s:%s (gRPC %s:%s) anonymously",
                    http_host, http_port, self._grpc_host, self._grpc_port,
                )
                client = weaviate.connect_to_local(
                    host=http_host,
                    port=http_port,
                    grpc_port=self._grpc_port,
                    additional_config=additional,
                )

            # Auto-provision the schema. Imported here to keep import-time deps
            # local to this module and avoid circular issues.
            from src.hybridrag.ingestion.ingestion_service.entities.weaviate_schema import (
                ensure_collection,
            )

            try:
                ensure_collection(client)
            except Exception:
                log.exception("WeaviateStore: ensure_collection failed; closing client")
                try:
                    client.close()
                except Exception:
                    pass
                raise

            self._client = client
            log.info(
                "WeaviateStore: connected in %.1f ms (collection=%s)",
                (time.perf_counter() - t0) * 1000.0,
                self._class_name,
            )
            return self._client

    def _collection(self) -> Any:
        """Return the active collection handle (connecting if needed)."""
        return self._connect().collections.get(self._class_name)

    # ------------------------------------------------------------- helpers
    async def _get_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        """Call the supplied embedding function, awaiting if it is async."""
        if inspect.iscoroutinefunction(self._embed):
            vecs = await self._embed(list(texts))
        else:
            vecs = self._embed(list(texts))
        # Coerce numpy / iterables into a plain list-of-list-of-floats for the
        # JSON-over-gRPC payload Weaviate expects.
        out: List[List[float]] = []
        for v in vecs:
            try:
                out.append([float(x) for x in v])
            except TypeError:
                # Single vector returned as scalar -> wrap it.
                out.append([float(v)])
        return out

    @staticmethod
    def _normalize_chunk(c: Union["Any", Dict[str, Any]]) -> Dict[str, Any]:
        """Normalise a Chunk dataclass or enriched dict to a property dict.

        Always returns a dict with at minimum: chunk_id, file_id, key, content.
        Unknown keys are filtered to the DocChunk schema. The raw text is
        returned alongside under the ``__text__`` private key so callers do not
        have to re-derive it for embedding.
        """
        if is_dataclass(c) and not isinstance(c, type):
            d = asdict(c)
            # Chunk has a derived chunk_id property; asdict() exposes _chunk_id
            # instead. Restore the canonical name.
            cid = d.pop("_chunk_id", None) or getattr(c, "chunk_id", None)
            if cid:
                d["chunk_id"] = cid
            # Chunk.text → schema's `content` field.
            if "text" in d and "content" not in d:
                d["content"] = d["text"]
        elif isinstance(c, dict):
            d = dict(c)
            # Prefer "content" but accept "text" for back-compat with Chunk.
            if "content" not in d and "text" in d:
                d["content"] = d["text"]
        else:
            raise TypeError(
                f"WeaviateStore: unsupported chunk type {type(c).__name__}; "
                "expected Chunk dataclass or dict"
            )

        text = d.get("content") or d.get("text") or ""
        # Keep only properties present in the schema. Anything else is dropped.
        props = {k: v for k, v in d.items() if k in _SCHEMA_PROPS and v is not None}
        # Required fields fallback.
        props.setdefault("chunk_id", d.get("chunk_id") or "")
        props.setdefault("file_id", d.get("file_id") or "")
        props.setdefault("key", d.get("key") or "")
        props.setdefault("content", text)
        props["__text__"] = text  # passthrough for embedding step
        return props

    @staticmethod
    def _build_filter(filters: Optional[Dict[str, Any]]) -> Any | None:
        """Translate a ``{prop: value}`` mapping into a v4 ``Filter`` chain.

        Multiple entries are AND-combined. Empty/None values are skipped.
        """
        if not filters:
            return None
        from weaviate.classes.query import Filter

        clauses = []
        for prop, value in filters.items():
            if value is None or value == "":
                continue
            if isinstance(value, (list, tuple, set)):
                clauses.append(Filter.by_property(prop).contains_any(list(value)))
            else:
                clauses.append(Filter.by_property(prop).equal(value))
        if not clauses:
            return None
        result = clauses[0]
        for clause in clauses[1:]:
            result = result & clause
        return result

    # --------------------------------------------------------- add_chunks
    async def precompute_embeddings(self, chunks: Iterable[Any]) -> None:
        """Pre-embed any chunks not already cached, so the next ``add_chunks``
        avoids the embedding network round-trip while it holds an index lock.

        Mirrors :meth:`FAISSStore.precompute_embeddings`.
        """
        norm = [self._normalize_chunk(c) for c in chunks]
        new = [n for n in norm if n["chunk_id"] and n["chunk_id"] not in self._embed_cache]
        if not new:
            return
        texts = [n["__text__"] for n in new]
        vecs = await self._get_embeddings(texts)
        for n, v in zip(new, vecs):
            self._embed_cache[n["chunk_id"]] = v
        log.info("WeaviateStore.precompute_embeddings: %d new vectors cached", len(new))

    async def add_chunks(self, chunks: Iterable[Any]) -> int:
        """Insert chunks into Weaviate using the v4 fixed-size batcher.

        Returns the number of objects sent for insertion. Existing chunk_ids
        are NOT re-checked here (Weaviate uses chunk_id-based UUIDs upstream
        in Phase 2B); callers should de-duplicate at the orchestrator level.
        """
        normalized = [self._normalize_chunk(c) for c in chunks]
        if not normalized:
            self._embed_cache.clear()
            return 0

        # Embed everything that wasn't pre-warmed.
        missing = [n for n in normalized if n["chunk_id"] not in self._embed_cache]
        if missing:
            texts = [n["__text__"] for n in missing]
            vecs = await self._get_embeddings(texts)
            for n, v in zip(missing, vecs):
                self._embed_cache[n["chunk_id"]] = v

        # Force-reconnect each call: persistent gRPC client in long-lived
        # worker corrupts multi-byte UTF-8 (Đ, đ, í etc) stored as
        # `�\udc90` pattern. Fresh client avoids the corruption.
        self.close()
        collection = self._collection()
        t0 = time.perf_counter()
        inserted = 0
        failed: list[tuple[str, str]] = []

        # Insert one object at a time via `data.insert` (REST/unary) instead
        # of `batch.fixed_size` / `insert_many` (both go through the gRPC
        # batch path). weaviate-client 4.18 + Weaviate 1.27's gRPC batch
        # corrupts multi-byte UTF-8 sequences on the server: bytes whose
        # first byte falls in 0xC0-0xDF (e.g. `Đ` = c4 90, `í` = c3 ad)
        # get re-decoded as cp1252 + surrogateescape and stored as
        # mojibake (`Ä\udc90`). The single-object `data.insert` path is
        # REST + JSON and round-trips Vietnamese diacritics losslessly.
        # Sync-loop, not asyncio.to_thread — weaviate-client v4 sync calls
        # re-enter the running asyncio loop, which raises "loop already
        # running" inside the worker's persistent loop.
        # Bypass weaviate-client entirely — use raw HTTP REST with
        # explicit UTF-8 to avoid the v4 client UTF-8 corruption bug
        # where multi-byte chars (Đ, í, ể) get re-decoded as cp1252.
        import json as _json
        import requests
        rest_url = f"{self._url.rstrip('/')}/v1/objects"
        # No session — fresh connection per request, no keep-alive (which
        # may be corrupting the worker's persistent HTTP/1.1 stream).
        common_headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Connection": "close",
        }
        if self._api_key:
            common_headers["Authorization"] = f"Bearer {self._api_key}"

        for n in normalized:
            cid = n["chunk_id"]
            vector = self._embed_cache.get(cid)
            if vector is None:
                log.warning("WeaviateStore.add_chunks: missing vector for %s", cid)
                continue
            props = {k: v for k, v in n.items() if not k.startswith("__")}
            body = _json.dumps(
                {
                    "class": self._class_name,
                    "properties": props,
                    "vector": list(vector),
                },
                ensure_ascii=False,
            ).encode("utf-8")
            if props.get("key", "").startswith("Đ"):
                log.warning(
                    "PRE-HTTP body_first_200_bytes=%r body_len=%d",
                    body[:200], len(body),
                )
            try:
                r = requests.post(rest_url, data=body, headers=common_headers, timeout=30)
                if r.status_code >= 300:
                    failed.append((cid, f"http {r.status_code}: {r.text[:200]}"))
                    log.warning(
                        "WeaviateStore.add_chunks: REST insert failed cid=%s status=%d body=%r",
                        cid, r.status_code, r.text[:200],
                    )
                    continue
                inserted += 1
            except Exception as exc:  # noqa: BLE001
                failed.append((cid, str(exc)))
                log.warning(
                    "WeaviateStore.add_chunks: insert failed cid=%s err=%s",
                    cid, exc,
                )

        self._embed_cache.clear()
        log.info(
            "WeaviateStore.add_chunks: inserted=%d failed=%d in %.1f ms",
            inserted,
            len(failed) if failed else 0,
            (time.perf_counter() - t0) * 1000.0,
        )
        return inserted

    # --------------------------------------------------------------- delete
    def delete_by_key(self, key: str) -> int:
        """Delete every object whose ``key`` property equals ``key``.

        Returns the number of objects removed (best-effort -- v4 returns a
        DeleteManyReturn with counts; fields differ across releases so we read
        whichever is available).
        """
        from weaviate.classes.query import Filter

        collection = self._collection()
        t0 = time.perf_counter()
        try:
            result = collection.data.delete_many(
                where=Filter.by_property("key").equal(key),
            )
        except Exception:
            log.exception("WeaviateStore.delete_by_key(%s) failed", key)
            return 0
        # v4's DeleteManyReturn exposes counts via these attributes (names have
        # been stable since 4.4 but we are defensive about it).
        successful = getattr(result, "successful", None)
        if successful is None:
            successful = getattr(result, "matches", 0)
        n = int(successful or 0)
        log.info(
            "WeaviateStore.delete_by_key: key=%s removed=%d in %.1f ms",
            key, n, (time.perf_counter() - t0) * 1000.0,
        )
        return n

    # --------------------------------------------------------------- exists
    def has_key(self, key: str) -> bool:
        """Return True iff at least one object with that ``key`` exists."""
        from weaviate.classes.query import Filter

        collection = self._collection()
        try:
            result = collection.query.fetch_objects(
                filters=Filter.by_property("key").equal(key),
                limit=1,
            )
        except Exception:
            log.exception("WeaviateStore.has_key(%s) failed", key)
            return False
        return bool(getattr(result, "objects", None))

    # ---------------------------------------------------------- health
    def health_check(self) -> Dict[str, Any]:
        """Probe Weaviate connectivity + collection state.

        Returns a status dict with: ``ok`` (bool), ``count`` (int, total
        objects in the collection), and ``error`` (str | None). Never
        raises; callers can log the dict at startup to verify the
        backend is reachable before serving requests.
        """
        from weaviate.classes.aggregate import GroupByAggregate  # noqa: F401  (sanity import)

        try:
            collection = self._collection()
            agg = collection.aggregate.over_all(total_count=True)
            count = int(getattr(agg, "total_count", 0) or 0)
            return {"ok": True, "count": count, "error": None}
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "WeaviateStore.health_check FAILED: %s: %s",
                type(exc).__name__, exc,
            )
            return {"ok": False, "count": 0, "error": f"{type(exc).__name__}: {exc}"}

    # -------------------------------------------------- batch fetch by id
    def get_by_chunk_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch DocChunk objects whose ``chunk_id`` property matches any of ``chunk_ids``.

        Uses a single ``fetch_objects`` round-trip with a ``contains_any``
        filter; the result dicts have the same shape as :meth:`search`
        output minus the ``score`` field. Used by the hybrid searcher to
        swap child hits for their full parent body (parent-document
        retrieval pattern).

        Returns an empty list on any error or empty input — callers fall
        back to the original (child) doc body so retrieval never blocks
        on a parent-fetch failure.
        """
        from weaviate.classes.query import Filter

        ids = [c for c in (chunk_ids or []) if c]
        if not ids:
            return []
        collection = self._collection()
        t0 = time.perf_counter()
        try:
            response = collection.query.fetch_objects(
                filters=Filter.by_property("chunk_id").contains_any(ids),
                limit=len(ids),
            )
        except Exception:
            log.exception("WeaviateStore.get_by_chunk_ids failed (n=%d)", len(ids))
            return []

        out: List[Dict[str, Any]] = []
        for obj in getattr(response, "objects", []) or []:
            props = dict(obj.properties or {})
            text = _recover_utf8(props.get("content") or "")
            out.append({
                "chunk_id":    props.get("chunk_id"),
                "file_id":     props.get("file_id"),
                "key":         _recover_utf8(props.get("key") or ""),
                "text":        text,
                "content":     text,
                "parent_id":   props.get("parent_id"),
                "header_path": props.get("header_path"),
                "section":     _recover_utf8(props.get("section") or ""),
                "chunk_level": props.get("chunk_level"),
                "is_table":    props.get("is_table"),
                "campus":      props.get("campus"),
                "doc_type":    props.get("doc_type"),
                "faculty":     props.get("faculty"),
                "major":       props.get("major"),
                "year":        props.get("year"),
            })
        log.info(
            "WeaviateStore.get_by_chunk_ids: requested=%d returned=%d in %.1f ms",
            len(ids), len(out), (time.perf_counter() - t0) * 1000.0,
        )
        return out

    # --------------------------------------------------------------- search
    async def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        filters: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None,
        vector_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run a Weaviate hybrid (BM25 + vector) search.

        Parameters
        ----------
        query : str
            Raw user query. Used for the BM25 side of the hybrid query
            and -- unless ``vector_query`` is provided -- also embedded
            for the vector side.
        top_k : int
            Maximum number of results to return.
        filters : dict | None
            Optional ``{property: value}`` mapping. Multiple keys are AND-ed.
            List values become a ``contains_any`` clause.
        alpha : float | None
            Hybrid weight; 0=BM25 only, 1=vector only. Falls back to
            ``settings.WEAVIATE_HYBRID_ALPHA``.
        vector_query : str | None
            Optional alternative text to embed for the vector side
            (e.g. a HyDE-generated hypothetical passage). When provided,
            BM25 still uses ``query``; the embedded vector comes from
            ``vector_query``.

        Returns
        -------
        list[dict]
            One dict per hit. Shape matches the existing FAISS/BM25 outputs
            with the union of metadata fields:
            ``chunk_id, file_id, key, text, content, score, parent_id,
            header_path, campus, year, ...``.
        """
        top_k = max(1, int(top_k))
        a = float(alpha if alpha is not None else settings.WEAVIATE_HYBRID_ALPHA)

        # Embed the query (vector side of the hybrid). Use ``vector_query``
        # when supplied so callers can plug in HyDE passages without
        # changing the BM25 query.
        embed_text = vector_query if (vector_query and vector_query.strip()) else query
        vecs = await self._get_embeddings([embed_text])
        qvec = vecs[0] if vecs else None

        from weaviate.classes.query import MetadataQuery

        collection = self._collection()
        flt = self._build_filter(filters)

        t0 = time.perf_counter()
        try:
            response = collection.query.hybrid(
                query=query,
                vector=qvec,
                alpha=a,
                limit=top_k,
                filters=flt,
                return_metadata=MetadataQuery(score=True),
            )
        except Exception as exc:
            # Surface the failure category in the log message at WARNING
            # so dashboards / log filters catch it without needing the
            # full traceback. Common causes: connection refused (Weaviate
            # down), schema missing, gRPC port mismatch, auth failure.
            log.warning(
                "WeaviateStore.search FAILED query=%r exc=%s: %s",
                query[:80], type(exc).__name__, exc,
            )
            log.exception("WeaviateStore.search traceback")
            return []

        out: List[Dict[str, Any]] = []
        for obj in getattr(response, "objects", []) or []:
            props = dict(obj.properties or {})
            score = 0.0
            meta = getattr(obj, "metadata", None)
            if meta is not None:
                score = float(getattr(meta, "score", 0.0) or 0.0)
            text = _recover_utf8(props.get("content") or "")
            hit = {
                "chunk_id":    props.get("chunk_id"),
                "file_id":     props.get("file_id"),
                "key":         _recover_utf8(props.get("key") or ""),
                "text":        text,        # back-compat with FAISS output
                "content":     text,        # explicit field used by retrievers
                "score":       score,
                "parent_id":   props.get("parent_id"),
                "header_path": props.get("header_path"),
                "section":     _recover_utf8(props.get("section") or ""),
                "chunk_level": props.get("chunk_level"),
                "is_table":    props.get("is_table"),
                "campus":      props.get("campus"),
                "doc_type":    props.get("doc_type"),
                "faculty":     props.get("faculty"),
                "major":       props.get("major"),
                "year":        props.get("year"),
            }
            out.append(hit)

        log.info(
            "WeaviateStore.search: query=%r top_k=%d alpha=%.2f hits=%d in %.1f ms",
            query, top_k, a, len(out), (time.perf_counter() - t0) * 1000.0,
        )
        return out

    # --------------------------------------------------------- shutdown
    def close(self) -> None:
        """Close the underlying Weaviate client, if any. Safe to call twice."""
        with self._client_lock:
            client = self._client
            self._client = None
        if client is None:
            return
        try:
            client.close()
            log.info("WeaviateStore: client closed")
        except Exception:
            log.exception("WeaviateStore: error while closing client")

    def __del__(self) -> None:
        # Best-effort cleanup; never raise from finaliser.
        try:
            self.close()
        except Exception:
            pass
