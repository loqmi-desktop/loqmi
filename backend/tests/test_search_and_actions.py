"""Tests for web search helpers and the ReAct action parser."""

from core.web_search import format_results
from core.inference import _parse_action


def test_format_results_empty():
    assert format_results([]) == "No results found."


def test_format_results_numbering():
    results = [
        {"title": "First", "url": "https://a.example", "snippet": "alpha"},
        {"title": "Second", "url": "https://b.example", "snippet": "beta"},
    ]
    text = format_results(results)
    assert "1. First" in text
    assert "https://a.example" in text
    assert "2. Second" in text


def test_parse_action_valid():
    action = _parse_action('<action>{"tool": "web_search", "query": "rust vs go"}</action>')
    assert action == {"tool": "web_search", "query": "rust vs go"}


def test_parse_action_with_leading_text():
    buffer = 'Let me look that up.\n<action>{"tool": "fetch_url", "url": "https://x.example"}</action>'
    action = _parse_action(buffer)
    assert action is not None
    assert action["tool"] == "fetch_url"


def test_parse_action_invalid_json_returns_none():
    assert _parse_action("<action>{not json}</action>") is None


def test_parse_action_no_tag_returns_none():
    assert _parse_action("just plain text") is None
