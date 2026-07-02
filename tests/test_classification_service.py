"""Service tests for services/classification_service.py — label matching + report."""
from unittest.mock import AsyncMock, MagicMock

import services.classification_service as cls
from schemas.chat import MessageClass
from services.classification_service import _match_class, classify_and_report

CLASSES = [MessageClass(id=1, name="Billing"), MessageClass(id=2, name="Technical Support")]


def test_match_exact_case_insensitive():
    assert _match_class("billing", CLASSES).id == 1


def test_match_fuzzy_close_name():
    assert _match_class("Technical Suport", CLASSES).id == 2


def test_match_returns_none_when_unrelated():
    assert _match_class("Completely Unrelated Thing", CLASSES) is None


async def test_reports_matched_class(monkeypatch):
    monkeypatch.setattr(cls, "_call_classification_llm", AsyncMock(return_value="Billing"))
    send = AsyncMock()
    monkeypatch.setattr(cls, "_send_classification", send)
    await classify_and_report(42, "why was I charged twice", CLASSES)
    send.assert_awaited_once_with(42, 1)


async def test_skips_report_when_unmatched(monkeypatch):
    monkeypatch.setattr(cls, "_call_classification_llm", AsyncMock(return_value="Gibberish"))
    send = AsyncMock()
    monkeypatch.setattr(cls, "_send_classification", send)
    await classify_and_report(42, "hello", CLASSES)
    send.assert_not_called()


async def test_swallows_errors(monkeypatch):
    monkeypatch.setattr(cls, "_call_classification_llm", AsyncMock(side_effect=RuntimeError("x")))
    # Must not raise — classification is best-effort background work.
    await classify_and_report(1, "q", CLASSES)


async def test_send_classification_puts_to_spring_boot(monkeypatch):
    http = MagicMock()
    response = MagicMock()
    http.put = AsyncMock(return_value=response)
    monkeypatch.setattr(cls, "get_http_client", lambda: http)
    await cls._send_classification(7, 3)
    http.put.assert_awaited_once()
    assert http.put.call_args.args[0].endswith("/internal/messages/7")
    response.raise_for_status.assert_called_once()
