from __future__ import annotations

from tools.web_search_tool import web_search_raw, format_web_results


def inject_web_context_into_state(
    state: dict,
    query: str,
    max_results: int = 3,
) -> dict:
    state = dict(state)

    try:
        results = web_search_raw(query=query, max_results=max_results)
        state["web_results"] = results
        state["web_query"] = query
        state["web_context"] = format_web_results(results)
    except Exception as e:
        state["web_results"] = []
        state["web_query"] = query
        state["web_context"] = f"Web search failed: {e}"

    return state