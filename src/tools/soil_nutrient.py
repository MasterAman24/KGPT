import json
from typing import Any, Dict
from .soil_gql_client import fetch_all_states, filter_by_state

# Cache for full-country fetch per cycle
_ALL_DATA_CACHE = None  # {"cycle": str, "data": [...]}

def soil_nutrient_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected input:
      state["tool_query"] = {
        "cycle": "2025-26",   # optional
        "state_name": "Bihar",
        "district_name": "Patna"   # optional (not filtered here, but you can extend)
      }
    """
    global _ALL_DATA_CACHE

    q = state.get("tool_query")
    if q is None:
        return {**state, "tool_result": {"error": "Empty query"}}

    if isinstance(q, str):
        try:
            q = json.loads(q)
        except Exception:
            return {**state, "tool_result": {"error": "Invalid query format"}}

    cycle = q.get("cycle", "2025-26")
    state_name = q.get("state_name")

    if not state_name:
        return {**state, "tool_result": {"error": "Missing 'state_name' in query"}}

    try:
        if _ALL_DATA_CACHE is None or _ALL_DATA_CACHE.get("cycle") != cycle:
            all_data = fetch_all_states(cycle)
            _ALL_DATA_CACHE = {"cycle": cycle, "data": all_data}
        else:
            all_data = _ALL_DATA_CACHE["data"]

        results = filter_by_state(all_data, state_name)
        if not results:
            return {**state, "tool_result": {"error": f"No data found for state '{state_name}'"}}

        return {
            **state,
            "tool_result": {
                "cycle": cycle,
                "state_name": state_name,
                "results": results
            }
        }

    except Exception as e:
        return {**state, "tool_result": {"error": str(e)}}
