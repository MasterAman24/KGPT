import requests
from bs4 import BeautifulSoup
from typing import Any, Dict

def mandi_price_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects tool_query: 'state,commodity' (e.g., 'Rajasthan,Wheat')
    """
    query = state.get("tool_query", "")
    if not query or "," not in str(query):
        return {**state, "tool_result": {"error": "tool_query must be 'state,commodity'"}}

    try:
        state_name, commodity = [p.strip() for p in str(query).split(",", 1)]

        base_url = "https://www.commodityonline.com/mandiprices/state"
        url = f"{base_url}/{state_name.lower().replace(' ', '-')}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-IN,en;q=0.9"
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rows = soup.select("tr")[1:]  # skip header
        results = []
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if not cols or len(cols) < 9:
                continue
            rec = {
                "Commodity": cols[0],
                "Arrival Date": cols[1],
                "Variety": cols[2],
                "State": cols[3],
                "District": cols[4],
                "Market": cols[5],
                "Min Price": cols[6],
                "Max Price": cols[7],
                "Avg Price": cols[8]
            }
            if rec["Commodity"].lower() == commodity.lower():
                results.append(rec)

        if not results:
            raise ValueError(f"No data found for commodity '{commodity}' in state '{state_name}'.")

        tool_result = {"results": results, "state": state_name, "commodity": commodity}

    except Exception as e:
        tool_result = {"error": str(e)}

    return {**state, "tool_result": tool_result}
