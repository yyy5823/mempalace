"""
test_closets.py — Tests for the closet (searchable index) layer.

Covers:
  * build_closet_lines — pointer-line shape, entity extraction, stoplist,
    quote/header pickup, and the "always emit one line" guarantee.
  * upsert_closet_lines — pure overwrite (no append), char-limit packing,
    atomic-line guarantee.
  * purge_file_closets — wipes prior closets so a re-mine starts clean.
  * The end-to-end rebuild: re-mining a file fully replaces its closets,
    including when the prior run produced more numbered closets.
  * search_memories closet-first path — returns chunk-level hits parsed
    from `→drawer_ids` pointers, falls back when closets are empty,
    respects max_distance.
"""

from mempalace.miner import mine
from mempalace.palace import (
    CLOSET_CHAR_LIMIT,
    build_closet_lines,
    get_closets_collection,
    purge_file_closets,
    upsert_closet_lines,
)
from mempalace.searcher import _extract_drawer_ids_from_closet, search_memories


# ── build_closet_lines ─────────────────────────────────────────────────


class TestBuildClosetLines:
    def test_emits_pointer_line_shape(self, tmp_path):
        content = (
            "# Auth rewrite\n\n"
            "Decided we need to migrate to passkeys. "
            "Built the prototype with WebAuthn. "
            "Reviewed the API surface."
        )
        lines = build_closet_lines(
            "/proj/auth.md",
            ["drawer_proj_backend_aaa", "drawer_proj_backend_bbb"],
            content,
            wing="proj",
            room="backend",
        )
        assert lines, "should always emit at least one line"
        for line in lines:
            assert "→" in line, f"line missing pointer arrow: {line!r}"
            parts = line.split("|")
            assert len(parts) == 3, f"expected topic|entities|→refs, got {line!r}"
            assert parts[2].startswith("→")

    def test_extracts_section_headers_as_topics(self):
        content = "# First Header\nbody\n## Second Header\nmore body"
        lines = build_closet_lines("/x.md", ["d1"], content, "w", "r")
        joined = "\n".join(lines).lower()
        assert "first header" in joined
        assert "second header" in joined

    def test_entity_stoplist_filters_sentence_starters(self):
        # "When", "After", "The" repeat 3+ times — old code would index them
        # as entities. New code's stoplist drops them.
        content = (
            "When the pipeline ran, the result was good. "
            "When the user logged in, the token was issued. "
            "After the migration, the latency dropped. "
            "After the rollback, the latency rose. "
            "The new flow is stable. The audit cleared."
        )
        lines = build_closet_lines("/x.md", ["d1"], content, "w", "r")
        # Entities sit between the two pipes
        entity_segments = [line.split("|")[1] for line in lines]
        for seg in entity_segments:
            tokens = set(seg.split(";")) if seg else set()
            assert "When" not in tokens
            assert "After" not in tokens
            assert "The" not in tokens

    def test_real_proper_nouns_survive_stoplist(self):
        content = (
            "Igor reviewed the diff. Milla wrote the spec. "
            "Igor pushed the fix. Milla approved the PR. "
            "Igor and Milla shipped together."
        )
        lines = build_closet_lines("/x.md", ["d1"], content, "w", "r")
        entity_segments = [line.split("|")[1] for line in lines]
        joined_entities = ";".join(entity_segments)
        assert "Igor" in joined_entities
        assert "Milla" in joined_entities

    def test_emits_fallback_line_when_nothing_extractable(self):
        # No headers, no action verbs, no quotes, no repeated capitalized words
        content = "lorem ipsum dolor sit amet consectetur adipiscing elit"
        lines = build_closet_lines("/x/notes.txt", ["d1"], content, "wing", "room")
        assert len(lines) == 1
        assert "wing/room/notes" in lines[0]
        assert "→d1" in lines[0]

    def test_pointer_references_first_three_drawers(self):
        ids = [f"drawer_{i}" for i in range(10)]
        lines = build_closet_lines("/x.md", ids, "# A\n# B", "w", "r")
        assert all("→drawer_0,drawer_1,drawer_2" in line for line in lines)


# ── upsert_closet_lines ───────────────────────────────────────────────


class TestUpsertClosetLines:
    def test_overwrites_existing_closet_does_not_append(self, palace_path):
        col = get_closets_collection(palace_path)
        base = "closet_test_room_abc"
        meta = {"wing": "test", "room": "room", "source_file": "/x.md"}

        # First mine — three short lines.
        upsert_closet_lines(col, base, ["alpha|;|→d1", "beta|;|→d2", "gamma|;|→d3"], meta)
        first = col.get(ids=[f"{base}_01"])
        assert "alpha" in first["documents"][0]
        assert "beta" in first["documents"][0]

        # Second mine — entirely different lines. Must replace, not append.
        upsert_closet_lines(col, base, ["delta|;|→d4", "epsilon|;|→d5"], meta)
        second = col.get(ids=[f"{base}_01"])
        doc = second["documents"][0]
        assert "delta" in doc
        assert "epsilon" in doc
        assert "alpha" not in doc, "old closet line leaked into rebuild"
        assert "beta" not in doc

    def test_packs_into_multiple_closets_without_splitting_lines(self, palace_path):
        col = get_closets_collection(palace_path)
        base = "closet_pack_room_def"
        meta = {"wing": "test", "room": "room", "source_file": "/y.md"}

        # Build lines that approach but never exceed the limit.
        line = "x" * 600  # well under CLOSET_CHAR_LIMIT
        n_written = upsert_closet_lines(col, base, [line, line, line, line], meta)
        # 4 lines @ 600+1 chars = 2404 — should pack into 2 closets (≤1500 each)
        assert n_written == 2

        for i in range(1, n_written + 1):
            doc = col.get(ids=[f"{base}_{i:02d}"])["documents"][0]
            # Every line is intact (never split mid-line)
            for chunk in doc.split("\n"):
                assert len(chunk) == 600, f"line was truncated in closet {i}"
            # Closet stays under the cap
            assert len(doc) <= CLOSET_CHAR_LIMIT


# ── purge_file_closets ────────────────────────────────────────────────


class TestPurgeFileClosets:
    def test_deletes_only_the_targeted_source(self, palace_path):
        col = get_closets_collection(palace_path)
        col.upsert(
            ids=["closet_a_01", "closet_b_01"],
            documents=["a|;|→d1", "b|;|→d2"],
            metadatas=[
                {"source_file": "/keep.md", "wing": "w", "room": "r"},
                {"source_file": "/drop.md", "wing": "w", "room": "r"},
            ],
        )
        purge_file_closets(col, "/drop.md")

        remaining_ids = set(col.get()["ids"])
        assert "closet_a_01" in remaining_ids
        assert "closet_b_01" not in remaining_ids


# ── End-to-end rebuild via the project miner ──────────────────────────


class TestMinerClosetRebuild:
    def test_remine_replaces_closets_completely(self, tmp_path):
        import yaml

        project = tmp_path / "proj"
        project.mkdir()
        (project / "mempalace.yaml").write_text(
            yaml.dump({"wing": "proj", "rooms": [{"name": "general", "description": "x"}]})
        )
        target = project / "doc.md"

        # First mine — long content produces multiple numbered closets.
        first_topics = "\n\n".join(f"# Topic {i}\n" + ("filler text " * 30) for i in range(15))
        target.write_text(first_topics)
        palace = tmp_path / "palace"
        mine(str(project), str(palace), wing_override="proj", agent="test")

        col = get_closets_collection(str(palace))
        first_pass = col.get(where={"source_file": str(target)})
        assert first_pass["ids"], "first mine should have written closets"
        first_ids = set(first_pass["ids"])
        assert any("topic 0" in (d or "").lower() for d in first_pass["documents"])

        # Touch mtime so file_already_mined doesn't short-circuit, and
        # rewrite with fewer topics (so the rebuild produces fewer closets
        # than the first run).
        import os
        import time

        target.write_text("# Only Topic Now\n" + ("short body " * 5))
        new_mtime = os.path.getmtime(target) + 60
        os.utime(target, (new_mtime, new_mtime))
        time.sleep(0.01)  # ensure mtime delta is visible

        mine(str(project), str(palace), wing_override="proj", agent="test")

        col = get_closets_collection(str(palace))
        second_pass = col.get(where={"source_file": str(target)})
        second_docs = "\n".join(second_pass["documents"]).lower()
        assert "only topic now" in second_docs
        for i in range(15):
            assert (
                f"topic {i}\n" not in second_docs
            ), f"stale 'Topic {i}' from first mine survived the rebuild"
        # Numbered closets that existed only in the larger first run must be gone.
        leftover = first_ids - set(second_pass["ids"])
        for stale_id in leftover:
            assert not col.get(ids=[stale_id])[
                "ids"
            ], f"orphan closet {stale_id} from larger first run survived purge"


# ── _extract_drawer_ids_from_closet ───────────────────────────────────


class TestExtractDrawerIds:
    def test_parses_single_pointer(self):
        assert _extract_drawer_ids_from_closet("topic|;|→drawer_x") == ["drawer_x"]

    def test_parses_multiple_pointers_per_line(self):
        line = "topic|ent|→drawer_a,drawer_b,drawer_c"
        assert _extract_drawer_ids_from_closet(line) == [
            "drawer_a",
            "drawer_b",
            "drawer_c",
        ]

    def test_dedupes_across_lines(self):
        doc = "one|;|→drawer_a,drawer_b\ntwo|;|→drawer_b,drawer_c"
        assert _extract_drawer_ids_from_closet(doc) == [
            "drawer_a",
            "drawer_b",
            "drawer_c",
        ]

    def test_empty_doc_returns_empty(self):
        assert _extract_drawer_ids_from_closet("") == []
        assert _extract_drawer_ids_from_closet("no arrows here") == []


# ── search_memories closet-first path ────────────────────────────────


class TestSearchMemoriesClosetFirst:
    def test_falls_back_to_direct_when_no_closets(self, palace_path, seeded_collection):
        # seeded_collection populates only mempalace_drawers, not closets.
        result = search_memories("JWT authentication", palace_path)
        assert result["results"], "should still find drawer hits via fallback"
        for hit in result["results"]:
            assert hit.get("matched_via") == "drawer"

    def test_closet_first_returns_chunk_level_hits(self, palace_path, seeded_collection):
        # Build a closet that points at the JWT drawer specifically.
        closets = get_closets_collection(palace_path)
        closets.upsert(
            ids=["closet_proj_backend_aaa_01"],
            documents=["JWT auth tokens|;|→drawer_proj_backend_aaa"],
            metadatas=[
                {
                    "wing": "project",
                    "room": "backend",
                    "source_file": "auth.py",
                }
            ],
        )

        result = search_memories("JWT authentication", palace_path)
        assert result["results"], "closet-first search should hydrate the drawer"
        top = result["results"][0]
        assert top["matched_via"] == "closet"
        # Must be the chunk-level drawer text, not a concatenation of every
        # drawer in the file.
        assert "JWT" in top["text"]
        assert (
            "Database migrations" not in top["text"]
        ), "closet path should not glue unrelated drawers together"
        assert "closet_preview" in top
        assert "→drawer_proj_backend_aaa" in top["closet_preview"]

    def test_max_distance_filters_closet_hits(self, palace_path, seeded_collection):
        closets = get_closets_collection(palace_path)
        closets.upsert(
            ids=["closet_proj_backend_aaa_01"],
            documents=["JWT auth tokens|;|→drawer_proj_backend_aaa"],
            metadatas=[
                {
                    "wing": "project",
                    "room": "backend",
                    "source_file": "auth.py",
                }
            ],
        )

        # max_distance=0.001 is essentially "must match exactly". The closet
        # path should reject everything and the caller falls back to direct
        # search (which also filters with the same threshold).
        result = search_memories(
            "completely unrelated query about quantum gardening",
            palace_path,
            max_distance=0.001,
        )
        # Either no results, or every result respected the threshold.
        for hit in result["results"]:
            assert hit["distance"] <= 0.001
