"""Unit tests for chat_service._extract_cited_chunks — citation parsing + cleanup."""
from services.chat_service import _extract_cited_chunks
from tests.factories import make_chunk


def test_extracts_cited_chunks_and_strips_markers():
    chunks = [
        make_chunk(youtube_video_id="a"),
        make_chunk(youtube_video_id="b"),
        make_chunk(youtube_video_id="c"),
    ]
    cited, clean = _extract_cited_chunks("Foo [1]. Bar [2, 3].", chunks)
    assert [c["youtube_video_id"] for c in cited] == ["a", "b", "c"]
    assert clean == "Foo. Bar."


def test_out_of_range_citations_ignored():
    cited, clean = _extract_cited_chunks("Answer [5].", [make_chunk()])
    assert cited == []
    assert clean == "Answer."


def test_arabic_comma_citation_parsed():
    chunks = [make_chunk(youtube_video_id="a"), make_chunk(youtube_video_id="b")]
    cited, _ = _extract_cited_chunks("جواب [1، 2]", chunks)
    assert len(cited) == 2


def test_duplicate_citations_deduped():
    cited, _ = _extract_cited_chunks("A [1] and again [1].", [make_chunk()])
    assert len(cited) == 1


def test_english_no_info_phrase_clears_citations():
    cited, clean = _extract_cited_chunks(
        "I don't have information about that [1].", [make_chunk()]
    )
    assert cited == []
    assert "[" not in clean


def test_arabic_no_info_phrase_clears_citations():
    cited, _ = _extract_cited_chunks("لا أملك معلومات عن ذلك [1]", [make_chunk()])
    assert cited == []
