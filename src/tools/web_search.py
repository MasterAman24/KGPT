from typing import Any, Dict
from ..tools.tavily_tool import tavily_search  # keep your existing tavily wrapper

def web_search_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state.get("tool_query")
    if not q:
        return {**state, "tool_result": {"error": "Empty query", "results": []}}
    try:
        result = tavily_search(str(q), max_results=6)
    except Exception as e:
        result = {"error": str(e), "results": []}
    return {**state, "tool_result": result}
