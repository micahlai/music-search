from __future__ import annotations

import pytest

from music_search.query_parser import parse_query


def test_v1_parser_preserves_normalized_semantic_prompt() -> None:
    query = parse_query("  dreamy guitar that slowly gets louder  ")

    assert query.semantic_text == "dreamy guitar that slowly gets louder"
    assert query.moods == ()


def test_parser_rejects_blank_prompt() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_query("   ")
