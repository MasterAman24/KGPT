import os
import requests

def tavily_search(query: str, max_results: int = 5):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set.")
    url = "https://api.tavily.com/search"
    payload = {"api_key": api_key, "query": query, "max_results": max_results}
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()
