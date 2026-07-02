"""Adapter tests for the small LLM task wrappers (query rewrite, title,
channel description). Each is driven by a mocked LLMClient."""
from unittest.mock import AsyncMock, MagicMock

from rag.llm.description_generator import DescriptionGenerator
from rag.llm.query_rewriter import QueryRewriter
from rag.llm.title_generator import TitleGenerator
from schemas.chat import ConversationMessage


async def test_rewrite_without_history_returns_query_unchanged():
    llm = MagicMock()
    llm.generate = AsyncMock()
    result = await QueryRewriter(llm).rewrite("standalone question", [])
    assert result == "standalone question"
    llm.generate.assert_not_called()


async def test_rewrite_with_history_resolves_via_llm():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=("  What is machine learning?  ", 5))
    history = [ConversationMessage(role="USER", content="tell me about ML")]
    result = await QueryRewriter(llm).rewrite("what is it", history)
    assert result == "What is machine learning?"


async def test_title_generator_strips_result():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=("  Intro to Machine Learning  ", 3))
    assert await TitleGenerator(llm).generate("q") == "Intro to Machine Learning"


async def test_title_generator_returns_none_when_blank():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=("   ", 3))
    assert await TitleGenerator(llm).generate("q") is None


async def test_title_generator_returns_none_on_error():
    llm = MagicMock()
    llm.generate = AsyncMock(side_effect=RuntimeError("provider down"))
    assert await TitleGenerator(llm).generate("q") is None


async def test_description_generator_uses_titles_and_excerpts():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=("A channel about cooking pasta.", 10))
    gen = DescriptionGenerator(llm)
    samples = [{"video_title": "Pasta 101", "excerpts": ["first boil the water"]}]
    result = await gen.regenerate(samples, "en")
    assert result == "A channel about cooking pasta."
    user_content = llm.generate.call_args.args[0][-1]["content"]
    assert "Pasta 101" in user_content
    assert "first boil the water" in user_content
