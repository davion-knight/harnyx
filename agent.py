from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace

from harnyx_miner_sdk.api import fetch_page, llm_chat, search_web
from harnyx_miner_sdk.decorators import entrypoint
from harnyx_miner_sdk.query import CitationRef, Query, Response

CHUTES_MODEL = "deepseek-ai/DeepSeek-V3.2-TEE"
SEARCH_PROVIDER = "desearch"
RESULTS_PER_QUERY = 5
MAX_EVIDENCE_ITEMS = 12
MAX_FOLLOW_UP_QUERIES = 3
MAX_PAGE_CONTENT_CHARS = 3000
LLM_TIMEOUT_SECONDS = 35.0
SEARCH_TIMEOUT_SECONDS = 25.0
FETCH_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class Evidence:
    index: int
    receipt_id: str
    result_id: str
    url: str
    title: str | None
    snippet: str | None


@entrypoint("query")
async def query(query: Query) -> Response:
    plan = await _plan(query.text)
    required_facts = plan["required_facts"]

    try:
        evidence = await _search(plan["search_queries"], start_index=1)
    except Exception:
        return Response(
            text=(
                "I could not retrieve search evidence for this question due to a "
                "search provider failure, so I cannot give a source-backed answer."
            )
        )

    missing = await _admit_evidence(query.text, required_facts, evidence)

    if missing and len(evidence) < MAX_EVIDENCE_ITEMS:
        try:
            follow_up_queries = [f"{query.text} {fact}" for fact in missing[:MAX_FOLLOW_UP_QUERIES]]
            more_evidence = await _search(follow_up_queries, start_index=len(evidence) + 1)
            evidence = evidence + more_evidence
            missing = await _admit_evidence(query.text, required_facts, evidence)
        except Exception:
            # The follow-up round is best-effort refinement, not required for a
            # valid answer -- a slow or failed provider here should not sink the
            # whole response when the first round already has usable evidence.
            pass

    try:
        answer, used_indices = await _synthesize(query.text, evidence, required_facts, missing)
    except Exception:
        return Response(
            text=(
                "I gathered search evidence but could not generate a synthesized "
                "answer due to a provider failure."
            )
        )

    citations = [
        CitationRef(receipt_id=item.receipt_id, result_id=item.result_id)
        for item in evidence
        if item.index in used_indices and item.snippet
    ]
    return Response(text=answer, citations=citations or None)


async def _plan(query_text: str) -> dict[str, list[str]]:
    """Enumerate the atomic facts required to answer, plus initial search queries.

    Doing this before searching prevents answering before the required facts are
    even named -- a known failure mode when the harness treats broad recall as
    equivalent to a complete answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research planner. Given a question, list the atomic facts "
                "needed to answer it correctly and 2-4 concrete web search queries that "
                "would find those facts. Watch for false premises, ambiguous entities, "
                "and date/version scope hidden in the question. Respond with strict JSON "
                'only: {"required_facts": ["..."], "search_queries": ["..."]}'
            ),
        },
        {"role": "user", "content": query_text},
    ]
    try:
        result = await llm_chat(
            provider="chutes",
            model=CHUTES_MODEL,
            messages=messages,
            temperature=0.0,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        data = _parse_json(result.llm.raw_text)
    except Exception:
        # Planning is an optimization, not a hard requirement -- if the
        # provider is briefly overloaded, fall back to treating the raw
        # question as both the fact to verify and the query to search,
        # instead of crashing the whole task.
        data = {}

    required_facts = [str(f).strip() for f in data.get("required_facts") or [] if str(f).strip()]
    search_queries = [str(q).strip() for q in data.get("search_queries") or [] if str(q).strip()]
    if not required_facts:
        required_facts = [query_text]
    if not search_queries:
        search_queries = [query_text]
    return {"required_facts": required_facts, "search_queries": search_queries[:4]}


async def _search(search_queries: Sequence[str], *, start_index: int) -> list[Evidence]:
    response = await search_web(
        tuple(search_queries),
        provider=SEARCH_PROVIDER,
        num=RESULTS_PER_QUERY,
        timeout=SEARCH_TIMEOUT_SECONDS,
    )
    evidence: list[Evidence] = []
    for offset, result in enumerate(response.results):
        if result.url is None:
            continue
        evidence.append(
            Evidence(
                index=start_index + offset,
                receipt_id=response.receipt_id,
                result_id=result.result_id,
                url=result.url,
                title=result.title,
                snippet=result.note,
            )
        )
    if not evidence:
        raise RuntimeError("search_web returned no usable evidence")
    return await _fetch_pages(evidence)


async def _fetch_pages(evidence: list[Evidence]) -> list[Evidence]:
    """Replace short search snippets with real page content where possible.

    Multi-hop, fact-heavy questions usually can't be answered from a ~200-char
    snippet alone -- fetching the actual page gives evidence-checking and
    synthesis something substantive to work from. A single slow or blocked
    page must not sink the others, so failures fall back to the snippet.
    """

    async def _fetch_one(item: Evidence) -> Evidence:
        try:
            result = await fetch_page(item.url, provider=SEARCH_PROVIDER, timeout=FETCH_TIMEOUT_SECONDS)
        except Exception:
            return item
        pages = result.response.data
        if not pages or not pages[0].content:
            return item
        content = pages[0].content.strip()[:MAX_PAGE_CONTENT_CHARS]
        if not content:
            return item
        return replace(item, snippet=content)

    return list(await asyncio.gather(*(_fetch_one(item) for item in evidence)))


async def _admit_evidence(
    query_text: str,
    required_facts: Sequence[str],
    evidence: Sequence[Evidence],
) -> list[str]:
    """Check each required fact against evidence scope, not just topical relevance.

    An official-looking source for the wrong entity, date, or version must not
    count as support -- this is the single most common way a plausible-looking
    answer turns out to be wrong.
    """
    evidence_block = _format_evidence(evidence)
    facts_block = "\n".join(f"- {fact}" for fact in required_facts)
    messages = [
        {
            "role": "system",
            "content": (
                "You check evidence against required facts for a research question. "
                "For each required fact, decide whether any evidence item actually "
                "supports it at the correct entity, date, and scope. Do not accept an "
                "official-looking source that covers the wrong entity, year, or version "
                "as support. Respond with strict JSON only: "
                '{"missing_facts": ["..."]} listing only facts with no valid supporting '
                "evidence."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query_text}\n\nRequired facts:\n{facts_block}\n\nEvidence:\n{evidence_block}",
        },
    ]
    try:
        result = await llm_chat(
            provider="chutes",
            model=CHUTES_MODEL,
            messages=messages,
            temperature=0.0,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        data = _parse_json(result.llm.raw_text)
    except Exception:
        # If the scope check itself fails, proceed with whatever evidence was
        # already found rather than crashing -- synthesis still runs the
        # false-premise/no-guessing rules over it.
        return []
    return [str(f).strip() for f in data.get("missing_facts") or [] if str(f).strip()]


async def _synthesize(
    query_text: str,
    evidence: Sequence[Evidence],
    required_facts: Sequence[str],
    missing_facts: Sequence[str],
) -> tuple[str, set[int]]:
    evidence_block = _format_evidence(evidence)
    missing_note = ""
    if missing_facts:
        missing_ratio = len(missing_facts) / max(len(required_facts), 1)
        if missing_ratio >= 0.5:
            # Most of what the question needs is ungrounded -- a full answer
            # would mostly be guessing, so refusal is the honest response.
            missing_note = (
                f"\n\nThese required facts have no supporting evidence: {', '.join(missing_facts)}. "
                "Most of what the question needs is unverified. State plainly that the "
                "question cannot be answered from the evidence instead of guessing."
            )
        else:
            # A minority of facts are unsupported -- refusing the whole answer
            # would throw away evidence that does support most of it.
            missing_note = (
                f"\n\nThese specific facts have no supporting evidence: {', '.join(missing_facts)}. "
                "Most of the required facts ARE supported below. Answer the question "
                "using that evidence, and explicitly flag only the listed facts as "
                "unverified rather than refusing to answer the whole question."
            )
    messages = [
        {
            "role": "system",
            "content": (
                "You write the final answer using only the numbered evidence provided. "
                "Synthesize the evidence into a direct answer -- never paste retrieved "
                "text or navigation content verbatim. Cite evidence with bracketed "
                "indices like [1]. Do not invent facts absent from the evidence. If the "
                "question rests on a false premise, correct it explicitly instead of "
                "answering the premise as asked. Answer as fully as the evidence "
                "supports -- do not refuse the entire question over one unverified detail."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query_text}\n\nEvidence:\n{evidence_block}{missing_note}\n\n"
                "Write a compact, direct answer with bracketed citations."
            ),
        },
    ]
    result = await llm_chat(
        provider="chutes",
        model=CHUTES_MODEL,
        messages=messages,
        temperature=0.2,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    text = result.llm.raw_text
    if not text:
        raise RuntimeError("chutes response missing assistant content")

    used_indices = {item.index for item in evidence if f"[{item.index}]" in text}
    if not used_indices:
        used_indices = {item.index for item in evidence}
    return text, used_indices


def _format_evidence(evidence: Sequence[Evidence]) -> str:
    return "\n".join(
        f"[{item.index}] {item.title or item.url} — {item.snippet or 'no snippet'} ({item.url})"
        for item in evidence
    )


def _parse_json(raw: str | None) -> dict:
    if not raw:
        return {}
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
