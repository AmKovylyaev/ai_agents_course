from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.
    """
    import httpx
    from bs4 import BeautifulSoup

    url = "https://duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        with httpx.Client(timeout=15, headers=headers, follow_redirects=True) as client:
            resp = client.get(url, params={"q": query})
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            results = []

            for r in soup.select(".result")[:max_results]:
                title_elem = r.select_one(".result__a")
                snippet_elem = r.select_one(".result__snippet")

                if title_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    results.append(f"{title}: {snippet}")

            return "\n\n".join(results) if results else "No results found."

    except Exception as e:
        return f"Search error: {e}"


def web_search_raw(query: str, max_results: int = 3) -> list[dict]:
    import httpx
    from bs4 import BeautifulSoup

    url = "https://duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    results = []

    with httpx.Client(timeout=15, headers=headers, follow_redirects=True) as client:
        resp = client.get(url, params={"q": query})
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for r in soup.select(".result")[:max_results]:
            title_elem = r.select_one(".result__a")
            snippet_elem = r.select_one(".result__snippet")
            href = title_elem.get("href", "") if title_elem else ""

            if title_elem:
                results.append(
                    {
                        "title": title_elem.get_text(strip=True),
                        "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                        "url": href,
                    }
                )

    return results


def format_web_results(results: list[dict], max_chars: int = 1200) -> str:
    if not results:
        return "No web results found."

    chunks = []
    for i, r in enumerate(results, start=1):
        block = "\n".join(
            [
                f"[Web Result #{i}]",
                f"Title: {r.get('title', '')}",
                f"Snippet: {r.get('snippet', '')}",
                f"URL: {r.get('url', '')}",
            ]
        )
        chunks.append(block)

    text = "\n\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text