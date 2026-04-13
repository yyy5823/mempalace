#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Semantic search against the palace.
Returns verbatim text — the actual words, never summaries.
"""

import logging
import re
from pathlib import Path

from .palace import get_closets_collection, get_collection

# Closet pointer line format: "topic|entities|→drawer_id_a,drawer_id_b"
# Multiple lines may join with newlines inside one closet document.
_CLOSET_DRAWER_REF_RE = re.compile(r"→([\w,]+)")

logger = logging.getLogger("mempalace_mcp")


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


def build_where_filter(wing: str = None, room: str = None) -> dict:
    """Build ChromaDB where filter for wing/room filtering."""
    if wing and room:
        return {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        return {"wing": wing}
    elif room:
        return {"room": room}
    return {}


def _extract_drawer_ids_from_closet(closet_doc: str) -> list:
    """Parse all `→drawer_id_a,drawer_id_b` pointers out of a closet document.

    Preserves order and dedupes.
    """
    seen: dict = {}
    for match in _CLOSET_DRAWER_REF_RE.findall(closet_doc):
        for did in match.split(","):
            did = did.strip()
            if did and did not in seen:
                seen[did] = None
    return list(seen.keys())


def _closet_first_hits(
    palace_path: str,
    query: str,
    where: dict,
    drawers_col,
    n_results: int,
    max_distance: float,
):
    """Run a closet-first search and return chunk-level drawer hits.

    Returns:
        non-empty list of hits when the closet path produced usable matches.
        ``None`` when the closet collection is empty/missing OR when every
        candidate drawer was filtered out (e.g. by max_distance); the
        caller should fall back to direct drawer search.
    """
    try:
        closets_col = get_closets_collection(palace_path, create=False)
    except Exception:
        return None

    try:
        ckwargs = {
            "query_texts": [query],
            "n_results": max(n_results * 2, 5),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            ckwargs["where"] = where
        closet_results = closets_col.query(**ckwargs)
    except Exception:
        return None

    closet_docs = closet_results["documents"][0] if closet_results["documents"] else []
    if not closet_docs:
        return None

    closet_metas = closet_results["metadatas"][0]
    closet_dists = closet_results["distances"][0]

    # Collect candidate drawer IDs in closet-rank order, dedupe, remember
    # which closet (and its distance/preview) introduced each one.
    drawer_id_order: list = []
    drawer_provenance: dict = {}
    for cdoc, cmeta, cdist in zip(closet_docs, closet_metas, closet_dists):
        for did in _extract_drawer_ids_from_closet(cdoc):
            if did in drawer_provenance:
                continue
            drawer_provenance[did] = (cdist, cdoc, cmeta)
            drawer_id_order.append(did)

    if not drawer_id_order:
        return None

    # Hydrate exactly those drawers — chunk-level, not whole-file.
    try:
        fetched = drawers_col.get(
            ids=drawer_id_order,
            include=["documents", "metadatas"],
        )
    except Exception:
        return None

    fetched_ids = fetched.get("ids") or []
    fetched_docs = fetched.get("documents") or []
    fetched_metas = fetched.get("metadatas") or []
    fetched_map = {
        did: (doc, meta) for did, doc, meta in zip(fetched_ids, fetched_docs, fetched_metas)
    }

    hits: list = []
    for did in drawer_id_order:
        if did not in fetched_map:
            continue  # closet pointed to a drawer that no longer exists
        doc, meta = fetched_map[did]
        cdist, cdoc, _ = drawer_provenance[did]
        if max_distance > 0.0 and cdist > max_distance:
            continue
        hits.append(
            {
                "text": doc,
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(meta.get("source_file", "?")).name,
                "similarity": round(max(0.0, 1 - cdist), 3),
                "distance": round(cdist, 4),
                "matched_via": "closet",
                "closet_preview": cdoc[:200],
            }
        )
        if len(hits) >= n_results:
            break

    return hits if hits else None


def search(query: str, palace_path: str, wing: str = None, room: str = None, n_results: int = 5):
    """
    Search the palace. Returns verbatim drawer content.
    Optionally filter by wing (project) or room (aspect).
    """
    try:
        col = get_collection(palace_path, create=False)
    except Exception:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        raise SearchError(f"No palace found at {palace_path}")

    where = build_where_filter(wing, room)

    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

    except Exception as e:
        print(f"\n  Search error: {e}")
        raise SearchError(f"Search error: {e}") from e

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        print(f'\n  No results found for: "{query}"')
        return

    print(f"\n{'=' * 60}")
    print(f'  Results for: "{query}"')
    if wing:
        print(f"  Wing: {wing}")
    if room:
        print(f"  Room: {room}")
    print(f"{'=' * 60}\n")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        similarity = round(max(0.0, 1 - dist), 3)
        source = Path(meta.get("source_file", "?")).name
        wing_name = meta.get("wing", "?")
        room_name = meta.get("room", "?")

        print(f"  [{i}] {wing_name} / {room_name}")
        print(f"      Source: {source}")
        print(f"      Match:  {similarity}")
        print()
        # Print the verbatim text, indented
        for line in doc.strip().split("\n"):
            print(f"      {line}")
        print()
        print(f"  {'─' * 56}")

    print()


def search_memories(
    query: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
    max_distance: float = 0.0,
) -> dict:
    """Programmatic search — returns a dict instead of printing.

    Used by the MCP server and other callers that need data.

    Args:
        query: Natural language search query.
        palace_path: Path to the ChromaDB palace directory.
        wing: Optional wing filter.
        room: Optional room filter.
        n_results: Max results to return.
        max_distance: Max cosine distance threshold. The palace collection uses
            cosine distance (hnsw:space=cosine) — 0 = identical, 2 = opposite.
            Results with distance > this value are filtered out. A value of
            0.0 disables filtering. Typical useful range: 0.3–1.0.
    """
    try:
        drawers_col = get_collection(palace_path, create=False)
    except Exception as e:
        logger.error("No palace found at %s: %s", palace_path, e)
        return {
            "error": "No palace found",
            "hint": "Run: mempalace init <dir> && mempalace mine <dir>",
        }

    where = build_where_filter(wing, room)

    # Closet-first search: scan the compact index, parse drawer pointers
    # from each matching line, then hydrate exactly those drawers. This
    # keeps the result shape chunk-level (consistent with direct search)
    # and applies the same max_distance filter.
    closet_hits = _closet_first_hits(
        palace_path=palace_path,
        query=query,
        where=where,
        drawers_col=drawers_col,
        n_results=n_results,
        max_distance=max_distance,
    )
    if closet_hits is not None:
        return {
            "query": query,
            "filters": {"wing": wing, "room": room},
            "total_before_filter": len(closet_hits),
            "results": closet_hits,
        }

    # Fallback: direct drawer search (no closets yet, or closets empty)
    try:
        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = drawers_col.query(**kwargs)
    except Exception as e:
        return {"error": f"Search error: {e}"}

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        # Filter on raw distance before rounding to avoid precision loss
        if max_distance > 0.0 and dist > max_distance:
            continue
        hits.append(
            {
                "text": doc,
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source_file": Path(meta.get("source_file", "?")).name,
                "similarity": round(max(0.0, 1 - dist), 3),
                "distance": round(dist, 4),
                "matched_via": "drawer",
            }
        )

    return {
        "query": query,
        "filters": {"wing": wing, "room": room},
        "total_before_filter": len(docs),
        "results": hits,
    }
