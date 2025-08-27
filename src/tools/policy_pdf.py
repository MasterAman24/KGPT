from typing import Any, Dict
from .vector_db import get_policy_vector_db

def policy_pdf_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state.get("tool_query", "")
    if not q:
        return {**state, "tool_result": {"error": "Empty query"}}
    try:
        policy_db = get_policy_vector_db()
        results = policy_db.similarity_search(str(q), k=5)
        result_list = [{"content": r.page_content, "metadata": r.metadata} for r in results]
        tool_result = {"results": result_list}
    except Exception as e:
        tool_result = {"error": str(e)}
    return {**state, "tool_result": tool_result}
