#!/usr/bin/env python3
"""
searcher.py — Find anything. Exact words.

Hybrid search: BM25 keyword matching + vector semantic similarity.
Searches closets first (fast index), then hydrates full drawer content.
Falls back to direct drawer search for palaces without closets.
"""

import logging
import math
import re
from pathlib import Path

from .palace import get_collection, get_closets_collection

logger = logging.getLogger("mempalace_mcp")


class SearchError(Exception):
    """Raised when search cannot proceed (e.g. no palace found)."""


def _bm25_score(query: str, document: str, k1: float = 1.5, b: float = 0.75, avg_dl: float = 500) -> float:
    """Simple BM25 score for a single document against a query.

    This is a lightweight keyword-matching signal that complements vector
    similarity. It catches exact matches that embeddings might miss
    (e.g., specific names, project codes, error messages).
    """
    query_terms = set(re.findall(r'\w{2,}', query.lower()))
    doc_terms = re.findall(r'\w{2,}', document.lower())
    if not query_terms or not doc_terms:
        return 0.0
    doc_len = len(doc_terms)
    term_freq = {}
    for t in doc_terms:
        term_freq[t] = term_freq.get(t, 0) + 1

    score = 0.0
    for term in query_terms:
        tf = term_freq.get(term, 0)
        if tf > 0:
            # Simplified IDF — treat each query term as moderately rare
            idf = math.log(2.0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_dl)
            score += idf * numerator / denominator
    return score


def _hybrid_rank(vector_results, query: str, vector_weight: float = 0.6, bm25_weight: float = 0.4):
    """Re-rank results using both vector distance and BM25 keyword score.

    Returns results sorted by combined score (higher = better).
    """
    if not vector_results:
        return vector_results

    # Normalize vector distances to 0-1 similarity
    max_dist = max(r.get("distance", 1.0) for r in vector_results) or 1.0
    for r in vector_results:
        vec_sim = max(0.0, 1 - r.get("distance", 1.0) / max(max_dist, 0.001))
        bm25 = _bm25_score(query, r.get("text", ""))
        # Normalize BM25 to roughly 0-1 range
        bm25_norm = min(bm25 / 3.0, 1.0)
        r["_hybrid_score"] = vector_weight * vec_sim + bm25_weight * bm25_norm
        r["bm25_score"] = round(bm25, 3)

    vector_results.sort(key=lambda r: r["_hybrid_score"], reverse=True)
    # Clean up internal field
    for r in vector_results:
        del r["_hybrid_score"]
    return vector_results


def build_where_filter(wing: str = None, room: str = None) -> dict:
    """Build ChromaDB where filter for wing/room filtering."""
    if wing and room:
        return {"$and": [{"wing": wing}, {"room": room}]}
    elif wing:
        return {"wing": wing}
    elif room:
        return {"room": room}
    return {}


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

    # Try closet-first search: search the compact index, then hydrate drawers
    closet_hits = []
    try:
        closets_col = get_closets_collection(palace_path, create=False)
        ckwargs = {
            "query_texts": [query],
            "n_results": n_results * 2,  # over-fetch closets to find best drawers
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            ckwargs["where"] = where
        closet_results = closets_col.query(**ckwargs)
        if closet_results["documents"][0]:
            closet_hits = list(zip(
                closet_results["documents"][0],
                closet_results["metadatas"][0],
                closet_results["distances"][0],
            ))
    except Exception:
        pass  # no closets yet — fall through to direct drawer search

    # If closets found results, hydrate the referenced drawers
    MAX_HYDRATION_CHARS = 10000  # cap to prevent blowup on large source files

    if closet_hits:
        import re
        seen_sources = set()
        hits = []
        for closet_doc, closet_meta, closet_dist in closet_hits:
            source = closet_meta.get("source_file", "")
            if source in seen_sources:
                continue
            seen_sources.add(source)

            # Find drawers for this source file, grep for most relevant chunk
            try:
                drawer_results = drawers_col.get(
                    where={"source_file": source},
                    include=["documents", "metadatas"],
                )
                if drawer_results.get("ids"):
                    # Drawer-grep: score each chunk against the query,
                    # return the best-matching chunk first + surrounding context
                    query_terms = set(re.findall(r'\w{2,}', query.lower()))
                    best_idx = 0
                    best_score = -1
                    for idx, doc in enumerate(drawer_results["documents"]):
                        doc_lower = doc.lower()
                        score = sum(1 for t in query_terms if t in doc_lower)
                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    # Build result: best chunk first, then neighbors
                    docs = drawer_results["documents"]
                    n_docs = len(docs)
                    # Include best chunk + 1 before + 1 after for context
                    start = max(0, best_idx - 1)
                    end = min(n_docs, best_idx + 2)
                    relevant_text = "\n\n".join(docs[start:end])

                    if len(relevant_text) > MAX_HYDRATION_CHARS:
                        relevant_text = relevant_text[:MAX_HYDRATION_CHARS] + f"\n\n[...truncated. {n_docs} total drawers. Use mempalace_get_drawer for full content.]"

                    meta = drawer_results["metadatas"][best_idx]
                    hits.append({
                        "text": relevant_text,
                        "wing": meta.get("wing", "unknown"),
                        "room": meta.get("room", "unknown"),
                        "source_file": Path(source).name,
                        "similarity": round(max(0.0, 1 - closet_dist), 3),
                        "distance": round(closet_dist, 4),
                        "matched_via": "closet",
                        "closet_preview": closet_doc[:200],
                        "drawer_index": best_idx,
                        "total_drawers": n_docs,
                    })
            except Exception:
                pass

            if len(hits) >= n_results:
                break

        if hits:
            # Re-rank with BM25 hybrid scoring
            hits = _hybrid_rank(hits, query)
            return {
                "query": query,
                "filters": {"wing": wing, "room": room},
                "total_before_filter": len(closet_hits),
                "results": hits,
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
            }
        )

    # Re-rank with BM25 hybrid scoring
    hits = _hybrid_rank(hits, query)
    return {
        "query": query,
        "filters": {"wing": wing, "room": room},
        "total_before_filter": len(docs),
        "results": hits,
    }
