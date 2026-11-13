"""
core/web_search.py
Web search + page fetch tools used by the ReAct tool-calling loop in inference.py.
DuckDuckGo scraping and SearXNG need no API key. Brave and Tavily require a
user-supplied key (stored via memory.set_search_key / memory.get_search_key).
"""

from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs 


from config import SEARCH_RESULT_COUNT, FETCH_MAX_CHARS, SEARXNG_INSTANCE_URL, SEARCH_REGION

try:
    import trafilatura
except ImportError:
    trafilatura = None

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; loqmiBot/1.0)"}


def search(query: str, method: str, brave_key: str | None = None, tavily_key: str | None = None) -> list[dict]:
    """Dispatch to the selected backend. Returns a list of {"title","url","snippet"}."""
    if method == "brave" and brave_key:
        return _search_brave(query, brave_key)
    if method == "tavily" and tavily_key:
        return _search_tavily(query, tavily_key)
    if method == "searxng":
        return _search_searxng(query)
    return _search_duckduckgo(query)


def format_results(results: list[dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}")
    return "\n\n".join(lines)


def fetch(url: str) -> str:
    """Fetch and clean the text content of a specific URL."""
    try:
        with httpx.Client(timeout=15, follow_redirects=True, headers=_HEADERS) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        return f"Failed to fetch URL: {e}"

    text = trafilatura.extract(html) if trafilatura else None
    if not text:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    if not text:
        return "Could not extract readable content from this page."

    return text[:FETCH_MAX_CHARS]


# ── Backends ──────────────────────────────────────────────────────────────────

def _unwrap_ddg(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    q = parse_qs(urlparse(href).query)
    return q["uddg"][0] if "uddg" in q else href


def _search_duckduckgo(query: str) -> list[dict]:
    try:
        from ddgs import DDGS
        with DDGS() as d:
            raw = d.text(query, region=SEARCH_REGION, max_results=SEARCH_RESULT_COUNT)
        results = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in raw
        ]
        logger.info("ddgs returned %d results for %r", len(results), query)
        return results
    except ImportError:
        return _search_duckduckgo_html(query)
    except Exception as e:
        # "No results found." usually means every backend was rate-limited or blocked.
        # The HTML scraper hits the same limit, so don't retry it here.
        logger.warning("ddgs found nothing for %r (%s)", query, e)
        return [{"title": "Search failed", "url": "", "snippet": f"search engines rate-limited or blocked the request ({e})"}]


def _search_duckduckgo_html(query: str) -> list[dict]:
    try:
        with httpx.Client(timeout=10, headers=_HEADERS) as client:
            resp = client.post("https://html.duckduckgo.com/html/", data={"q": query})
            logger.info("DDG HTML status=%s", resp.status_code)
            resp.raise_for_status()
    except Exception as e:
        logger.warning("DuckDuckGo request failed: %s", e)
        return [{"title": "Search failed", "url": "", "snippet": str(e)}]

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result in soup.select(".result")[:SEARCH_RESULT_COUNT]:
        title_el = result.select_one(".result__title a")
        snippet_el = result.select_one(".result__snippet")
        if not title_el:
            continue
        results.append({
            "title": title_el.get_text(strip=True),
            "url": _unwrap_ddg(title_el.get("href", "")),
            "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
        })
    logger.info("DDG HTML parsed %d results for %r", len(results), query)
    return results


def fetch_text(url: str, max_chars: int = FETCH_MAX_CHARS) -> str | None:
    """Like fetch(), but returns None on any failure instead of an error string."""
    try:
        with httpx.Client(timeout=12, follow_redirects=True, headers=_HEADERS) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                return None
            html = resp.text
    except Exception as e:
        logger.info("fetch_text failed for %s: %s", url, e)
        return None

    text = trafilatura.extract(html) if trafilatura else None
    if not text:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    return text[:max_chars] if text else None


def _search_searxng(query: str) -> list[dict]:
    try:
        with httpx.Client(timeout=10, headers=_HEADERS) as client:
            resp = client.get(
                f"{SEARXNG_INSTANCE_URL}/search",
                params={"q": query, "format": "json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return [{"title": "Search failed", "url": "", "snippet": str(e)}]

    results = []
    for r in data.get("results", [])[:SEARCH_RESULT_COUNT]:
        results.append({"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")})
    return results


def _search_brave(query: str, api_key: str) -> list[dict]:
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": SEARCH_RESULT_COUNT},
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return [{"title": "Search failed", "url": "", "snippet": str(e)}]

    results = []
    for r in data.get("web", {}).get("results", [])[:SEARCH_RESULT_COUNT]:
        results.append({"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")})
    return results


def _search_tavily(query: str, api_key: str) -> list[dict]:
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": SEARCH_RESULT_COUNT},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return [{"title": "Search failed", "url": "", "snippet": str(e)}]

    results = []
    for r in data.get("results", [])[:SEARCH_RESULT_COUNT]:
        results.append({"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")})
    return results