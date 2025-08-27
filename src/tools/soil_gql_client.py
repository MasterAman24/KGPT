import requests
from typing import Any, Dict

GQL_URL = "https://soilhealth4.dac.gov.in/"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://soilhealth.dac.gov.in",
    "Referer": "https://soilhealth.dac.gov.in/",
    "Accept": "*/*",
}

GQL_QUERY = """
query GetNutrientDashboardForPortal(
  $state: ID, $district: ID, $block: ID, $village: ID,
  $cycle: String, $count: Boolean
) {
  getNutrientDashboardForPortal(
    state: $state
    district: $district
    block: $block
    village: $village
    cycle: $cycle
    count: $count
  )
}
"""

def gql_post(query: str, variables: Dict[str, Any]):
    payload = {
        "operationName": "GetNutrientDashboardForPortal",
        "variables": variables,
        "query": query,
    }
    r = requests.post(GQL_URL, json=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()
    if "errors" in j:
        raise RuntimeError(j["errors"])
    return j["data"]["getNutrientDashboardForPortal"]

def fetch_all_states(cycle: str = "2025-26"):
    return gql_post(GQL_QUERY, {"cycle": cycle})

def filter_by_state(all_data, state_name: str):
    state_name = (state_name or "").lower()
    return [
        row for row in all_data
        if str(row.get("state", {}).get("name", "")).strip().lower() == state_name
    ]
