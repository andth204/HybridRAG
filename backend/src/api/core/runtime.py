from __future__ import annotations
import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Union
from openai import AsyncOpenAI
from src.config.prompts import SAFETY_RULES, get_prompt
from src.config.settings import settings
from src.hybridrag.chat.answer import AnswerGenerator
from src.hybridrag.chat.clarifier import Clarifier
from src.hybridrag.chat.session_state import SessionStateRepo
from src.hybridrag.chat.slot_filler import SlotFiller
from src.hybridrag.chat.tools import execute_tool, tools_for_intent
from src.hybridrag.retrieval.hybrid import HybridSearcher
from src.hybridrag.retrieval.weaviate_hybrid import WeaviateHybridSearcher
from src.hybridrag.router import KeywordRouter, ROUTES, SemanticRouter
from src.hybridrag.router.intent_router import IntentRouter
from src.hybridrag.utils.prompt_security import sanitize_user_text

log = logging.getLogger(__name__)

WARMUP_QUERIES_FILE = Path(settings.DATA_DIR) / "eval" / "warmup_queries.txt"


def compact_sources(items: list[dict[str, Any]], limit: int = 5) -> list[str]:
    names: list[str] = []
    for item in items:
        name = (
            item.get("title")
            or item.get("source")
            or item.get("key")
            or item.get("id")
            or "unknown"
        )
        value = str(name).strip()
        if value and value not in names:
            names.append(value)
        if len(names) >= limit:
            break
    return names


def get_router_type() -> str:
    configured = str(getattr(settings, "ROUTER_TYPE", "keyword")).strip().lower()
    return "semantic" if configured == "semantic" else "keyword"


@lru_cache(maxsize=1)
def get_hybrid_searcher() -> HybridSearcher:
    searcher = HybridSearcher()
    searcher.load_indexes()
    return searcher


@lru_cache(maxsize=1)
def get_weaviate_searcher() -> WeaviateHybridSearcher:
    """Cached Weaviate-backed hybrid searcher (Phase 2 alternative path).

    Lazily constructed only when ``settings.RETRIEVAL_BACKEND="weaviate"``.
    The underlying Weaviate client is itself lazy (opens on the first
    ``search`` call) so importing this module never touches the network.
    """
    searcher = WeaviateHybridSearcher()
    searcher.load_indexes()
    return searcher


def get_active_searcher() -> Union[HybridSearcher, WeaviateHybridSearcher]:
    """Return whichever retrieval backend is active per
    ``settings.RETRIEVAL_BACKEND``. Default (``"hybrid_legacy"``) returns
    the legacy FAISS+BM25 searcher so production behaviour is unchanged.
    """
    backend = str(getattr(settings, "RETRIEVAL_BACKEND", "hybrid_legacy")).strip().lower()
    if backend == "weaviate":
        return get_weaviate_searcher()
    return get_hybrid_searcher()


@lru_cache(maxsize=1)
def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator()


@lru_cache(maxsize=1)
def get_keyword_router() -> KeywordRouter:
    return KeywordRouter(routes=ROUTES)


@lru_cache(maxsize=1)
def get_semantic_router() -> SemanticRouter:
    return SemanticRouter(
        routes=ROUTES,
        embeddings_dir=settings.ROUTER_EMBEDDINGS_DIR,
    )


# -------------------------------------------------------------------- #
# Phase 4 — dialogue layer caches (intent / slot / clarifier).
#
# All four caches are cheap to populate (no network, no DB) so it is
# safe to call them from a synchronous warm-up path. The SessionState
# repo defers all I/O to the first read/write — constructing it does
# not open a Postgres connection.
# -------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_intent_router() -> IntentRouter:
    """Cached fine-grained intent router (keyword-first + semantic fallback).

    The semantic side is lazily instantiated on the first ambiguous query
    so cold-starts stay cheap when the keyword bank is sufficient.
    """
    return IntentRouter()


@lru_cache(maxsize=1)
def get_clarifier() -> Clarifier:
    """Cached clarification policy gate (stateless after construction)."""
    return Clarifier()


@lru_cache(maxsize=1)
def get_session_state_repo() -> SessionStateRepo:
    """Cached session-state repository bound to ``settings.DATABASE_URL``.

    Construction is pure (no connection borrow); the first read/write
    inside ``borrow()`` will lazily build the pooled connection.
    """
    return SessionStateRepo(settings.DATABASE_URL)


@lru_cache(maxsize=1)
def get_slot_filler() -> SlotFiller:
    """Cached slot filler wired to the shared session-state repo."""
    return SlotFiller(repo=get_session_state_repo())


# -------------------------------------------------------------------- #
# Tool-call branch (Phase 4D)
#
# When the intent classifier picks a structured-store intent
# (score_lookup / tuition_lookup / program_info / compare), the
# pipeline takes a two-call detour:
#
#   1. First LLM call advertises the tool schemas + the rewritten user
#      query. The model decides whether to invoke a tool.
#   2. If it does, we execute the tool synchronously off the event loop
#      (Postgres queries) and feed the result back into a SECOND LLM
#      call together with the RAG context the question already
#      assembled. The second call streams to the user.
#
# The helper returns the final assembled answer text plus the structured
# tool-call audit log (used by chat.py for the assistant-message
# metadata).
# -------------------------------------------------------------------- #
SYSTEM_PROMPT_TOOL_CALL = (
    "You are a Vietnamese admissions advisor for Hung Yen University "
    "of Technology and Education. When the user asks for specific "
    "numbers (admission scores / tuition / list of majors), CALL THE "
    "MATCHING TOOL instead of guessing. After the tool returns, "
    "combine its data with any retrieval context to write a clear, "
    "natural Vietnamese answer. If the tool returns an empty result, "
    "refuse honestly (e.g. 'Tôi chưa có dữ liệu điểm chuẩn của ...'). "
    "Cite source file names from CONTEXT in a final "
    "[Thong tin tham chieu] section when factual claims come from RAG."
)


def _record_completion_usage(response: Any, *, feature: str) -> None:
    """Best-effort: persist completion token usage to chat_cost_log."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        from src.hybridrag.utils.cost_tracker import record as _cost_record
        _cost_record(
            model=settings.GENERATE_MODEL,
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
            feature=feature,
        )
    except Exception:
        log.debug("cost record failed", exc_info=True)


def _summarise_tool_result(result_json: str, *, head: int = 3) -> dict[str, Any]:
    """Build a compact audit summary of one tool call's result.

    The full payload is intentionally NOT mirrored into chat metadata —
    we only persist a row count plus the first ``head`` entries so a
    misbehaving tool can't bloat the messages table.
    """
    try:
        parsed = json.loads(result_json) if result_json else {}
    except Exception:  # noqa: BLE001
        return {"row_count": 0, "preview": [], "error": "unparseable"}

    if not isinstance(parsed, dict):
        return {"row_count": 0, "preview": [], "error": "non_object"}

    if "error" in parsed:
        return {"row_count": 0, "preview": [], "error": parsed["error"]}

    data = parsed.get("data")
    if isinstance(data, list):
        return {"row_count": len(data), "preview": data[:head]}
    if data is None:
        return {"row_count": 0, "preview": []}
    return {"row_count": 1, "preview": [data]}


async def generate_with_tools(
    *,
    rewritten_query: str,
    intent: str,
    retrieved_docs: list[dict[str, Any]],
    session_slots: dict[str, Any] | None = None,
    timeout: float = 25.0,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Run the tool-call branch and return (answer_text, tool_calls, used_tool).

    Behaviour contract:

    * If no tool schemas exist for ``intent``, signal the caller (via
      ``used_tool=False`` and an empty answer) to fall back to RAG.
    * If the first LLM call returns ``tool_calls`` we execute each one
      via :func:`execute_tool` (wrapped in ``asyncio.to_thread`` because
      the underlying lookup is sync Postgres I/O), then make a second
      LLM call that streams the final answer.
    * If any tool returns an ``error`` envelope, the helper still
      returns ``used_tool=False`` so the caller can fall through to
      plain RAG. The error envelope is **not** surfaced to the user.
    * If the first LLM call returns no ``tool_calls`` (model decided
      not to call), ``used_tool=False`` and answer is empty — caller
      should run RAG.

    The ``tool_calls`` audit log is always returned (empty on the
    fall-back paths) so chat.py can persist it on the assistant message.
    """
    tools = tools_for_intent(intent)
    if not tools:
        return "", [], False

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    # Phase 5.4 follow-up — sanitize the rewritten query before it is
    # embedded in the user message. The query reaches this helper after
    # the rewriter, which itself accepts arbitrary user input, so a
    # malicious chat-template token or close-tag could otherwise survive
    # all the way to the first tool-LLM call.
    safe_rewritten = sanitize_user_text(rewritten_query)

    # Build a slot-aware user message so the model has the conversation
    # state without re-asking. Display values (not canonical keys) so the
    # prompt reads like natural Vietnamese.
    slot_context = ""
    if session_slots:
        bits: list[str] = []
        for name in ("major", "year", "campus", "faculty"):
            val = session_slots.get(name)
            if val is not None and val != "":
                bits.append(f"{name}={val}")
        if bits:
            slot_context = "\n[Slot context]: " + "; ".join(bits)

    # Wrap the user query in a trust-boundary tag — same XML envelope
    # the RAG/chitchat prompts use — so the model can structurally tell
    # data from instructions inside the user role too.
    user_content = (
        f"<user_question>\n{safe_rewritten}\n</user_question>"
        f"{slot_context}"
    ).strip()

    first_messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT_TOOL_CALL},
        {"role": "user", "content": user_content},
    ]

    try:
        first_response = await client.chat.completions.create(
            model=settings.GENERATE_MODEL,
            messages=first_messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=settings.MAX_GEN_MAIN,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("generate_with_tools: first LLM call failed: %s", exc)
        return "", [], False

    _record_completion_usage(first_response, feature="tool_call_first")

    if not first_response.choices:
        return "", [], False

    first_msg = first_response.choices[0].message
    raw_tool_calls = getattr(first_msg, "tool_calls", None) or []

    # Audit log (regardless of whether we end up using results).
    tool_call_log: list[dict[str, Any]] = []

    if not raw_tool_calls:
        # Model declined to call a tool → caller should run RAG.
        return "", [], False

    # 2) Execute every tool call sequentially. The Postgres lookups are
    #    fast (~ms) so parallelism is not worth the complexity here.
    tool_messages: list[dict[str, Any]] = []
    any_error = False
    for call in raw_tool_calls:
        fn = call.function
        name = fn.name
        args_json = fn.arguments or "{}"
        try:
            # Wrap sync lookup_*() in to_thread so we never block the
            # event loop while Postgres is doing its work.
            result_json = await asyncio.to_thread(execute_tool, name, args_json)
        except Exception as exc:  # noqa: BLE001 - defensive
            log.warning("generate_with_tools: execute_tool(%s) raised: %s", name, exc)
            result_json = json.dumps(
                {"error": "tool_runtime_error", "detail": str(exc)},
                ensure_ascii=False,
            )

        summary = _summarise_tool_result(result_json)
        tool_call_log.append(
            {
                "name": name,
                "arguments": args_json,
                "result_summary": summary,
            }
        )
        if summary.get("error"):
            any_error = True
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result_json,
            }
        )

    if any_error:
        # Per spec: surface nothing to the user, let chat.py fall back to RAG.
        log.warning("generate_with_tools: at least one tool returned an error envelope")
        return "", tool_call_log, False

    # 3) Second LLM call — feed the tool results + retrieval context
    #    back to the model, no tools this time, stream the final answer.
    context_block = ""
    if retrieved_docs:
        try:
            context_block = get_answer_generator().build_context(retrieved_docs)
        except Exception as exc:  # noqa: BLE001
            log.warning("generate_with_tools: build_context failed: %s", exc)
            context_block = ""

    second_messages: list[dict[str, Any]] = list(first_messages)
    # Append the assistant's tool_call turn — required by OpenAI's
    # protocol so the subsequent role=tool messages link back.
    second_messages.append(
        {
            "role": "assistant",
            "content": first_msg.content or "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments or "{}",
                    },
                }
                for c in raw_tool_calls
            ],
        }
    )
    second_messages.extend(tool_messages)
    if context_block:
        second_messages.append(
            {
                "role": "system",
                "content": f"CONTEXT (RAG):\n{context_block}",
            }
        )

    try:
        second_response = await client.chat.completions.create(
            model=settings.GENERATE_MODEL,
            messages=second_messages,
            temperature=settings.TEMPERATURE_MAIN,
            max_tokens=settings.MAX_GEN_MAIN,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("generate_with_tools: second LLM call failed: %s", exc)
        return "", tool_call_log, False

    _record_completion_usage(second_response, feature="tool_call_second")

    answer = ""
    if second_response.choices:
        answer = (second_response.choices[0].message.content or "").strip()

    if not answer:
        return "", tool_call_log, False
    return answer, tool_call_log, True


def _load_warmup_queries(path: Path = WARMUP_QUERIES_FILE) -> list[str]:
    if not path.exists():
        log.info("Warm-up queries file not found at %s - skipping pre-warm", path)
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Failed to read warm-up queries file %s: %s", path, exc)
        return []
    queries: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    return queries


async def _prewarm_embedding_cache(
    searcher: Union[HybridSearcher, WeaviateHybridSearcher],
    queries: list[str],
) -> None:
    if not queries:
        return
    # The legacy HybridSearcher has a VectorSearcher with an LRU embedding
    # cache; route through it directly so cache hits on real requests are
    # immediate. The Weaviate searcher does not own such a cache (Weaviate
    # itself handles vector storage), so we just warm the OpenAI embedder.
    vector_searcher = getattr(searcher, "vector", None)
    if vector_searcher is None or not hasattr(vector_searcher, "_embed"):
        # Fallback: warm OpenAI side only; LRU on VectorSearcher will not be
        # filled - real requests will pay one extra embed each. This is the
        # expected path for the Weaviate backend.
        log.info(
            "No VectorSearcher._embed reachable for pre-warm; "
            "warming the embedder client directly"
        )
        from src.hybridrag.ingestion.embedding import embedder
        await embedder.embed(queries)
        return
    await vector_searcher._embed(queries)


def warm_up_runtime() -> None:
    # Warm up whichever retrieval backend is active.
    searcher = get_active_searcher()

    # Surface Weaviate connectivity at startup. A silent connection
    # failure here used to manifest as every chat query returning the
    # canned refusal — easier to spot at boot than at request time.
    if isinstance(searcher, WeaviateHybridSearcher):
        try:
            searcher.load_indexes()
            store = getattr(searcher, "_store", None)
            if store is not None and hasattr(store, "health_check"):
                report = store.health_check()
                if report.get("ok"):
                    log.info(
                        "Weaviate health: OK (collection objects=%d)",
                        report.get("count", 0),
                    )
                else:
                    log.error(
                        "Weaviate health: FAILED (%s). Retrieval will "
                        "return empty results until the backend is "
                        "reachable.",
                        report.get("error"),
                    )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Weaviate health probe raised (%s); continuing startup",
                exc,
            )

    if get_router_type() == "semantic":
        get_semantic_router()
    else:
        get_keyword_router()

    # Phase 4: warm the dialogue-layer caches. None of these calls do
    # network or DB I/O — the IntentRouter only loads keyword YAML, the
    # Clarifier is stateless, the SlotFiller is a pure repo wrapper, and
    # the SessionStateRepo lazily borrows the pool on the first read.
    try:
        get_intent_router()
        get_clarifier()
        get_slot_filler()  # also primes get_session_state_repo()
    except Exception as exc:  # noqa: BLE001 - never block startup on warmup
        log.warning("Dialogue-layer warm-up failed (non-fatal): %s", exc)

    if not bool(getattr(settings, "WARMUP_QUERIES_ENABLED", True)):
        log.info("Embedding pre-warm disabled via WARMUP_QUERIES_ENABLED=False")
        return

    try:
        queries = _load_warmup_queries()
        if not queries:
            return
        # warm_up_runtime() is called from a worker thread via asyncio.to_thread,
        # so the current thread has no running event loop -> asyncio.run is safe.
        asyncio.run(_prewarm_embedding_cache(searcher, queries))
        log.info("Pre-warmed embedding cache with %d queries", len(queries))
    except Exception as exc:
        # Pre-warm is best-effort - never block startup on it.
        log.warning("Embedding pre-warm failed (non-fatal): %s", exc)


# Vietnamese system message for the chitchat branch. Pins the assistant
# to the admissions advisor persona AND embeds the same SAFETY_RULES
# block the RAG branch uses, so the chitchat path is no longer the easy
# prompt-injection target it used to be (cf. Phase 5.4 follow-up).
_CHITCHAT_SYSTEM_MESSAGE = (
    "Bạn là trợ lý tư vấn tuyển sinh tiếng Việt thân thiện của Trường Đại học "
    "Sư phạm Kỹ thuật Hưng Yên (UTEHY). Bạn chỉ trao đổi xã giao ngắn gọn và "
    "định hướng người hỏi quay lại các chủ đề tuyển sinh / chương trình / "
    "học phí / phương thức xét tuyển.\n\n"
    f"{SAFETY_RULES}"
)


async def stream_chitchat_answer(query: str, timeout: float = 20.0) -> AsyncIterator[str]:
    """Stream a chitchat reply, sanitizing the user query first.

    The chitchat prompt template wraps the user text in
    ``<user_question>...</user_question>``, and the system role embeds
    ``SAFETY_RULES`` so the model treats everything inside the wrapper as
    DATA. The sanitizer also defangs chat-template tokens (``<|im_start|>``,
    ``[INST]``, …) and HTML-escapes the angle brackets so the user
    cannot close the wrapping tag from inside.
    """
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    safe_query = sanitize_user_text(query)
    prompt = get_prompt("answer_generation_chitchat", query=safe_query)
    stream = await client.chat.completions.create(
        model=settings.GENERATE_MODEL,
        messages=[
            {"role": "system", "content": _CHITCHAT_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=settings.TEMPERATURE_CHITCHAT,
        max_tokens=settings.MAX_GEN_CHITCHAT,
        stream=True,
        stream_options={"include_usage": True},
        timeout=timeout,
    )
    usage_in = 0
    usage_out = 0
    try:
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_in = int(chunk.usage.prompt_tokens or 0)
                usage_out = int(chunk.usage.completion_tokens or 0)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta or not delta.content:
                continue
            yield delta.content
    finally:
        if usage_in or usage_out:
            try:
                from src.hybridrag.utils.cost_tracker import record as _cost_record
                _cost_record(
                    model=settings.GENERATE_MODEL,
                    tokens_in=usage_in,
                    tokens_out=usage_out,
                    feature="chitchat",
                )
            except Exception:
                pass