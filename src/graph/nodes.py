
import os
import re
import json
import requests
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from langchain.schema import SystemMessage, HumanMessage
from ..llm.groq_client import make_llm
from ..tools.tavily_tool import tavily_search

import sys

# Force Chroma to use a modern SQLite version from pysqlite3-binary
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass


# ====== Vector DB for PDFs ======
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PDF_FOLDER = "Major Schemes"
VECTOR_DB_DIR = "vector_db_policy"

def get_policy_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(VECTOR_DB_DIR):
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

    docs = []
    if os.path.isdir(PDF_FOLDER):
        for file in os.listdir(PDF_FOLDER):
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
                docs.extend(loader.load())

    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_DIR)
    vectordb.persist()
    return vectordb

policy_db = get_policy_vector_db()

# ====== Helper: India states / UTs (for mandi parsing) ======
INDIA_STATES_UTS = {
    "andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh","goa","gujarat","haryana","himachal pradesh",
    "jharkhand","karnataka","kerala","madhya pradesh","maharashtra","manipur","meghalaya","mizoram","nagaland",
    "odisha","punjab","rajasthan","sikkim","tamil nadu","telangana","tripura","uttar pradesh","uttarakhand","west bengal",
    "andaman and nicobar islands","chandigarh","dadra and nagar haveli and daman and diu","delhi","lakshadweep",
    "puducherry","jammu and kashmir","ladakh","nct of delhi","ncr"
}

# ====== Tools ======

# Web Search (Tavily)
def web_search_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("tool_query"):
        return {**state, "tool_result": {"error": "Empty query"}}
    try:
        result = tavily_search(state["tool_query"], max_results=6)
    except Exception as e:
        result = {"error": str(e), "results": []}
    return {**state, "tool_result": result}

# Mandi Price (scrape)
def mandi_price_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects tool_query: 'state,commodity' (e.g., 'Rajasthan,Wheat')
    """
    query = state.get("tool_query", "")
    if not query or "," not in query:
        return {**state, "tool_result": {"error": "tool_query must be 'state,commodity'"}}

    try:
        state_name, commodity = [p.strip() for p in query.split(",", 1)]

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

# Weather (WeatherAPI)
#WEATHERAPI_KEY=os.getenv("OPENWEATHER_API_KEY")
WEATHERAPI_KEY="75c43d92e1f8407590b205917251108"


def weather_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state.get("tool_query")
    if not q:
        return {**state, "tool_result": {"error": "Empty query"}}
    if not WEATHERAPI_KEY:
        return {**state, "tool_result": {"error": "Missing WEATHERAPI_KEY"}}

    city = str(q).strip()  # already formatted to a city name by formatter
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q={city}&aqi=no"

    try:
        res = requests.get(url, timeout=15).json()
        if "error" in res:
            raise ValueError(res["error"].get("message", "Error fetching weather"))
        weather_info = {
            "location": f"{res['location']['name']}, {res['location']['country']}",
            "localtime": res['location']['localtime'],
            "temperature_c": res['current']['temp_c'],
            "temperature_f": res['current']['temp_f'],
            "feels_like_c": res['current']['feelslike_c'],
            "condition": res['current']['condition']['text'],
            "humidity": res['current']['humidity'],
            "wind_kph": res['current']['wind_kph'],
            "wind_dir": res['current']['wind_dir'],
        }
    except Exception as e:
        weather_info = {"error": str(e)}

    return {**state, "tool_result": weather_info}

# Policy PDF Search (Chroma)
def policy_pdf_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state.get("tool_query", "")
    if not q:
        return {**state, "tool_result": {"error": "Empty query"}}
    try:
        results = policy_db.similarity_search(q, k=5)
        result_list = [{"content": r.page_content, "metadata": r.metadata} for r in results]
        tool_result = {"results": result_list}
    except Exception as e:
        tool_result = {"error": str(e)}
    return {**state, "tool_result": tool_result}

import requests
import json
from typing import Dict, Any

GQL_URL = "https://soilhealth4.dac.gov.in/"

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

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://soilhealth.dac.gov.in",
    "Referer": "https://soilhealth.dac.gov.in/",
    "Accept": "*/*",
}

# ---------------------------
# Core Functions
# ---------------------------

def gql_post(query: str, variables: dict):
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

def fetch_all_states(cycle="2025-26"):
    """Fetch entire India dataset once (all states/districts)."""
    return gql_post(GQL_QUERY, {"cycle": cycle})

def filter_by_state(all_data, state_name: str):
    """Filter dataset by state name (case-insensitive)."""
    state_name = state_name.lower()
    return [
        row for row in all_data
        if str(row.get("state", {}).get("name", "")).strip().lower() == state_name
    ]

# ---------------------------
# Tool Node
# ---------------------------

# Cache so we don’t hit API every call
_ALL_DATA_CACHE = None

def soil_nutrient_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool node to get soil nutrient info for a given state.
    Expected input:
      state["tool_query"] = {
        "cycle": "2025-26",   # optional, default = 2025-26
        "state_name": "Bihar"
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
        # Fetch all data once per cycle (cache)
        if _ALL_DATA_CACHE is None or _ALL_DATA_CACHE.get("cycle") != cycle:
            all_data = fetch_all_states(cycle)
            _ALL_DATA_CACHE = {"cycle": cycle, "data": all_data}
        else:
            all_data = _ALL_DATA_CACHE["data"]

        # Filter by state
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



# ====== Argument formatting helpers ======
def _extract_city_for_weather(text: str) -> str:
    text_l = text.lower()
    # Common patterns: "weather in Delhi", "temperature in Mumbai today"
    m = re.search(r"(?:in|at)\s+([a-zA-Z .'-]+)", text_l)
    if m:
        city = m.group(1).strip(" .'-")
        # cut trailing words like 'today', 'now'
        city = re.split(r"\b(today|now|tomorrow|this week|forecast)\b", city)[0].strip()
        return city.title()
    # Fallback: last capitalized token heuristic
    tokens = re.findall(r"[A-Z][a-zA-Z'-]+", text)
    if tokens:
        return tokens[-1]
    return text.strip()

def _extract_mandi_state_commodity(text: str) -> Tuple[str, str]:
    t = text.lower()
    # Find state/UT
    found_state = None
    for st in INDIA_STATES_UTS:
        if st in t:
            found_state = st
            break
    # Commodity: after "price of|rate of|for" or last word
    m = re.search(r"(?:price|prices|rate|rates)\s+(?:of|for)\s+([a-zA-Z /'-]+)", t)
    commodity = None
    if m:
        commodity = m.group(1).strip(" .'-")
    if not commodity:
        # Try words before 'in <state>'
        if found_state:
            before = t.split(found_state)[0]
            last = re.findall(r"[a-z]+", before)
            if last:
                commodity = last[-1]
    if not found_state:
        # Last resort: Delhi/NCR alias
        if "delhi" in t or "ncr" in t:
            found_state = "nct of delhi"
    if not commodity:
        commodity = "wheat"
    if not found_state:
        found_state = "rajasthan"
    return (found_state.title(), commodity.title())


def decide_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Plans which tools to call and RETURNS ALREADY-FORMATTED tool_query per tool.
    """
    llm = make_llm()
    system = """
You are a routing & argument-formatting controller.

TOOLS & ARG FORMAT:
- web_search: tool_query = natural language search string.
- weather:    tool_query = city/town name ONLY (e.g., "Delhi", "Mumbai", "Patna").
- policy_pdf: tool_query = short keyword query for government schemes/policies (e.g., "PMFBY insurance coverage").
- mandi_price:tool_query = "STATE,COMMODITY" (exactly one comma). Example: "Rajasthan,Wheat".
- soil_nutrient: tool_query = JSON object with keys: cycle (YYYY-YY), state_name, district_name (optional).
                 Example: {"cycle":"2025-26","state_name":"Bihar"}
                 Example with district: {"cycle":"2025-26","state_name":"Bihar","district_name":"Patna"}

Return JSON ONLY:
{
  "need_tool": bool,
  "tools_to_call": [
    {"tool_name": "weather", "tool_query": "Delhi"},
    {"tool_name": "mandi_price", "tool_query": "Rajasthan,Wheat"}
  ]
}
Pick ALL tools that are required to fully answer the user.
"""
    user_q = state.get("english_input") or state.get("user_input") or ""
    msgs = [SystemMessage(content=system), HumanMessage(content=f"User question: {user_q}")]
    out = llm.invoke(msgs).content.strip()

    # Defaults
    need_tool = False
    tools_to_call: List[Dict[str, Any]] = []

    try:
        data = json.loads(out)
        need_tool = bool(data.get("need_tool", False))
        tools_to_call = data.get("tools_to_call", [])
    except Exception:
        # Fallback coarse routing if LLM JSON fails
        txt = (user_q or "").lower()
        if any(k in txt for k in ["weather", "temperature", "rain", "forecast"]):
            need_tool = True
            tools_to_call.append({"tool_name": "weather", "tool_query": _extract_city_for_weather(user_q)})
        if any(k in txt for k in ["mandi", "market price", "crop price", "vegetable price", "commodity price"]):
            need_tool = True
            st, com = _extract_mandi_state_commodity(user_q)
            tools_to_call.append({"tool_name": "mandi_price", "tool_query": f"{st},{com}"})
        if any(k in txt for k in ["policy", "scheme", "act", "government"]):
            need_tool = True
            tools_to_call.append({"tool_name": "policy_pdf", "tool_query": user_q})
        if any(k in txt for k in ["soil", "nutrient", "soil health", "fertility", "nitrogen", "phosphorus", "potassium"]):
            need_tool = True
            # crude extraction of state name from query text
            found_state = None
            for st in INDIA_STATES_UTS:
                if st.lower() in txt:
                    found_state = st
                    break
            payload = {"cycle": "2025-26"}
            if found_state:
                payload["state_name"] = found_state
            tools_to_call.append({"tool_name": "soil_nutrient", "tool_query": payload})
        if any(k in txt for k in ["latest", "news", "update"]):
            need_tool = True
            tools_to_call.append({"tool_name": "web_search", "tool_query": user_q})

    # Post-process / normalize arg formats
    normalized: List[Dict[str, Any]] = []
    for t in tools_to_call:
        name = (t.get("tool_name") or "").strip()
        q = t.get("tool_query", "")

        if name == "weather":
            city = _extract_city_for_weather(str(q or user_q))
            normalized.append({"tool_name": "weather", "tool_query": city or "Delhi"})

        elif name == "mandi_price":
            if isinstance(q, str) and "," in q:
                state_part, commodity_part = [p.strip() for p in q.split(",", 1)]
            else:
                state_part, commodity_part = _extract_mandi_state_commodity(user_q)
            normalized.append({"tool_name": "mandi_price", "tool_query": f"{state_part},{commodity_part}"})

        elif name == "policy_pdf":
            normalized.append({"tool_name": "policy_pdf", "tool_query": str(q or user_q)})

        elif name == "web_search":
            normalized.append({"tool_name": "web_search", "tool_query": str(q or user_q)})

        elif name == "soil_nutrient":
            # Ensure dict JSON with allowed keys
            payload = {"cycle": "2025-26", "state_name": None, "district_name": None}
            if isinstance(q, dict):
                payload.update({k: q.get(k, payload[k]) for k in payload})
            elif isinstance(q, str):
                try:
                    j = json.loads(q)
                    payload.update({k: j.get(k, payload[k]) for k in payload})
                except Exception:
                    pass
            normalized.append({"tool_name": "soil_nutrient", "tool_query": payload})

    # Deduplicate tools
    seen = set()
    deduped = []
    for t in normalized:
        key = (t["tool_name"], json.dumps(t["tool_query"], sort_keys=True) if isinstance(t["tool_query"], dict) else t["tool_query"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    print("📝 decide_tool_node decision:")
    print("   Input:", user_q)
    print("   LLM raw:", out)
    print("   Final tools_to_call:", deduped)

    return {**state, "need_tool": bool(deduped), "tools_to_call": deduped}





# ====== Multi-tool executor (concurrent, single state update) ======
def _run_single_tool(tool_name: str, query: Any, base_state: Dict[str, Any]) -> Dict[str, Any]:
    tool_map = {
        "web_search": web_search_tool_node,
        "weather": weather_tool_node,
        "policy_pdf": policy_pdf_tool_node,
        "mandi_price": mandi_price_tool_node,
        "soil_nutrient": soil_nutrient_tool_node,
    }
    fn = tool_map.get(tool_name)
    if not fn:
        return {"tool": tool_name, "query": query, "output": {"error": f"Unknown tool: {tool_name}"}}
    try:
        tool_state = fn({**base_state, "tool_query": query})
        output = tool_state.get("tool_result") or tool_state.get("soil_nutrient_result") or {}
        return {"tool": tool_name, "query": query, "output": output}
    except Exception as e:
        return {"tool": tool_name, "query": query, "output": {"error": str(e)}}

def multi_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    plans = state.get("tools_to_call", [])
    if not plans:
        return state

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(plans)))) as ex:
        futures = [ex.submit(_run_single_tool, p["tool_name"], p["tool_query"], state) for p in plans]
        for fut in as_completed(futures):
            results.append(fut.result())

    # Maintain stable order similar to input plans
    ordered = []
    for p in plans:
        for r in results:
            if r["tool"] == p["tool_name"] and r["query"] == p["tool_query"]:
                ordered.append(r); break

    return {**state, "tool_results": ordered}

# ====== Answer node (merges typed outputs) ======
def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = make_llm()

    # Build evidence text from labeled outputs
    parts: List[str] = []
    for item in state.get("tool_results", []):
        tname = item.get("tool")
        q = item.get("query")
        out = item.get("output", {})

        if tname == "weather":
            if "error" in out:
                parts.append(f"[weather] ({q}) ERROR: {out['error']}")
            else:
                parts.append(
                    f"[weather] {out.get('location')} at {out.get('localtime')} → "
                    f"{out.get('temperature_c')}°C, {out.get('condition')} "
                    f"(feels {out.get('feels_like_c')}°C, humidity {out.get('humidity')}%, wind {out.get('wind_kph')} kph {out.get('wind_dir')})"
                )

        elif tname == "mandi_price":
            if "error" in out:
                parts.append(f"[mandi_price] ({q}) ERROR: {out['error']}")
            else:
                lines = []
                for r in out.get("results", []):
                    lines.append(
                        f"{r.get('Arrival Date','—')} • {r.get('Market','—')} • Min {r.get('Min Price','—')} | "
                        f"Avg {r.get('Avg Price','—')} | Max {r.get('Max Price','—')}"
                    )
                head = f"[mandi_price] {out.get('state','?')}, {out.get('commodity','?')}:\n" if lines else "[mandi_price] No rows."
                parts.append(head + ("\n".join(lines) if lines else ""))

        elif tname == "policy_pdf":
            if "error" in out:
                parts.append(f"[policy_pdf] ({q}) ERROR: {out['error']}")
            else:
                for r in out.get("results", []):
                    src = (r.get("metadata", {}) or {}).get("source", "unknown")
                    preview = (r.get("content") or "").strip().replace("\n", " ")
                    if len(preview) > 300:
                        preview = preview[:300] + "..."
                    parts.append(f"[policy_pdf] {src}: {preview}")

        elif tname == "web_search":
            if "error" in out:
                parts.append(f"[web_search] ({q}) ERROR: {out['error']}")
            else:
                for r in out.get("results", []):
                    title = r.get("title","")
                    url = r.get("url","")
                    snip = (r.get("content","") or "").strip().replace("\n"," ")
                    if len(snip) > 240: snip = snip[:240] + "..."
                    parts.append(f"[web_search] {title} — {snip} (Source: {url})")

        elif tname == "soil_nutrient":
            if "error" in out:
                parts.append(f"[soil_nutrient] ({q}) ERROR: {out['error']}")
            else:
                parts.append(f"[soil_nutrient] params={out.get('level')} data={str(out.get('results'))[:400]}...")

        else:
            parts.append(f"[{tname}] {json.dumps(out, ensure_ascii=False)[:400]}...")

    context = "\n\n".join(parts)
    user_q = state.get("english_input") or state.get("user_input") or ""
    prompt = f"User question: {user_q}\n\nAvailable evidence (may be partial):\n{context}\n\nCompose a concise, actionable answer. If data is missing, say what is missing and suggest how to get it."

    msgs = [SystemMessage(content="You are Krishi GPT, a farmer's helper which uses different tools attached to you and provide short solutions."), HumanMessage(content=prompt)]
    out = llm.invoke(msgs).content.strip()

    return {**state, "final_answer": out}

# import json
# from typing import Any, Dict, List

# from langchain.schema import SystemMessage, HumanMessage

# # Your existing LLM factory (unchanged path if you keep it)
# from ..llm.groq_client import make_llm

# # Tools
# from src.tools.web_search import web_search_tool_node
# from src.tools.weather import weather_tool_node
# from src.tools.policy_pdf import policy_pdf_tool_node
# from src.tools.mandi_price import mandi_price_tool_node
# from src.tools.soil_nutrient import soil_nutrient_tool_node
# from src.tools.utils import extract_city_for_weather, extract_mandi_state_commodity
# from src.tools.config import INDIA_STATES_UTS

# # ========== Tool Router ==========
# def decide_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Plans which tools to call and RETURNS ALREADY-FORMATTED tool_query per tool.
#     """
#     llm = make_llm()
#     system = """
# You are a routing & argument-formatting controller.

# TOOLS & ARG FORMAT:
# - web_search: tool_query = natural language search string.
# - weather:    tool_query = city/town name ONLY (e.g., "Delhi", "Mumbai", "Patna").
# - policy_pdf: tool_query = short keyword query for government schemes/policies (e.g., "PMFBY insurance coverage").
# - mandi_price:tool_query = "STATE,COMMODITY" (exactly one comma). Example: "Rajasthan,Wheat".
# - soil_nutrient: tool_query = JSON object with keys: cycle (YYYY-YY), state_name, district_name (optional).
#                  Example: {"cycle":"2025-26","state_name":"Bihar"}
#                  Example with district: {"cycle":"2025-26","state_name":"Bihar","district_name":"Patna"}

# Return JSON ONLY:
# {
#   "need_tool": bool,
#   "tools_to_call": [
#     {"tool_name": "weather", "tool_query": "Delhi"},
#     {"tool_name": "mandi_price", "tool_query": "Rajasthan,Wheat"}
#   ]
# }
# Pick ALL tools that are required to fully answer the user.
# """
#     user_q = state.get("english_input") or state.get("user_input") or ""
#     msgs = [SystemMessage(content=system), HumanMessage(content=f"User question: {user_q}")]
#     out = llm.invoke(msgs).content.strip()

#     need_tool = False
#     tools_to_call: List[Dict[str, Any]] = []

#     # LLM-first, heuristic fallback
#     try:
#         data = json.loads(out)
#         need_tool = bool(data.get("need_tool", False))
#         tools_to_call = data.get("tools_to_call", [])
#     except Exception:
#         txt = (user_q or "").lower()
#         if any(k in txt for k in ["weather", "temperature", "rain", "forecast"]):
#             need_tool = True
#             tools_to_call.append({"tool_name": "weather", "tool_query": extract_city_for_weather(user_q)})
#         if any(k in txt for k in ["mandi", "market price", "crop price", "vegetable price", "commodity price"]):
#             need_tool = True
#             st, com = extract_mandi_state_commodity(user_q)
#             tools_to_call.append({"tool_name": "mandi_price", "tool_query": f"{st},{com}"})
#         if any(k in txt for k in ["policy", "scheme", "act", "government"]):
#             need_tool = True
#             tools_to_call.append({"tool_name": "policy_pdf", "tool_query": user_q})
#         if any(k in txt for k in ["soil", "nutrient", "soil health", "fertility", "nitrogen", "phosphorus", "potassium"]):
#             need_tool = True
#             found_state = None
#             for st in INDIA_STATES_UTS:
#                 if st.lower() in txt:
#                     found_state = st
#                     break
#             payload = {"cycle": "2025-26"}
#             if found_state:
#                 payload["state_name"] = found_state
#             tools_to_call.append({"tool_name": "soil_nutrient", "tool_query": payload})
#         if any(k in txt for k in ["latest", "news", "update"]):
#             need_tool = True
#             tools_to_call.append({"tool_name": "web_search", "tool_query": user_q})

#     # Normalize formats
#     normalized: List[Dict[str, Any]] = []
#     for t in tools_to_call:
#         name = (t.get("tool_name") or "").strip()
#         q = t.get("tool_query", "")

#         if name == "weather":
#             city = extract_city_for_weather(str(q or user_q))
#             normalized.append({"tool_name": "weather", "tool_query": city or "Delhi"})

#         elif name == "mandi_price":
#             if isinstance(q, str) and "," in q:
#                 state_part, commodity_part = [p.strip() for p in q.split(",", 1)]
#             else:
#                 state_part, commodity_part = extract_mandi_state_commodity(user_q)
#             normalized.append({"tool_name": "mandi_price", "tool_query": f"{state_part},{commodity_part}"})

#         elif name == "policy_pdf":
#             normalized.append({"tool_name": "policy_pdf", "tool_query": str(q or user_q)})

#         elif name == "web_search":
#             normalized.append({"tool_name": "web_search", "tool_query": str(q or user_q)})

#         elif name == "soil_nutrient":
#             payload = {"cycle": "2025-26", "state_name": None, "district_name": None}
#             if isinstance(q, dict):
#                 payload.update({k: q.get(k, payload[k]) for k in payload})
#             elif isinstance(q, str):
#                 try:
#                     j = json.loads(q)
#                     payload.update({k: j.get(k, payload[k]) for k in payload})
#                 except Exception:
#                     pass
#             normalized.append({"tool_name": "soil_nutrient", "tool_query": payload})

#     # Deduplicate
#     seen = set()
#     deduped = []
#     for t in normalized:
#         key = (t["tool_name"], json.dumps(t["tool_query"], sort_keys=True) if isinstance(t["tool_query"], dict) else t["tool_query"])
#         if key not in seen:
#             seen.add(key)
#             deduped.append(t)

#     print("📝 decide_tool_node decision:")
#     print("   Input:", user_q)
#     print("   LLM raw:", out)
#     print("   Final tools_to_call:", deduped)

#     return {**state, "need_tool": bool(deduped), "tools_to_call": deduped}

# # ========== Multi-tool executor ==========
# def _run_single_tool(tool_name: str, query: Any, base_state: Dict[str, Any]) -> Dict[str, Any]:
#     tool_map = {
#         "web_search": web_search_tool_node,
#         "weather": weather_tool_node,
#         "policy_pdf": policy_pdf_tool_node,
#         "mandi_price": mandi_price_tool_node,
#         "soil_nutrient": soil_nutrient_tool_node,
#     }
#     fn = tool_map.get(tool_name)
#     if not fn:
#         return {"tool": tool_name, "query": query, "output": {"error": f"Unknown tool: {tool_name}"}}
#     try:
#         tool_state = fn({**base_state, "tool_query": query})
#         output = tool_state.get("tool_result") or {}
#         return {"tool": tool_name, "query": query, "output": output}
#     except Exception as e:
#         return {"tool": tool_name, "query": query, "output": {"error": str(e)}}

# from concurrent.futures import ThreadPoolExecutor, as_completed

# def multi_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     plans = state.get("tools_to_call", [])
#     if not plans:
#         return state

#     results: List[Dict[str, Any]] = []
#     with ThreadPoolExecutor(max_workers=min(8, max(1, len(plans)))) as ex:
#         futures = [ex.submit(_run_single_tool, p["tool_name"], p["tool_query"], state) for p in plans]
#         for fut in as_completed(futures):
#             results.append(fut.result())

#     ordered = []
#     for p in plans:
#         for r in results:
#             if r["tool"] == p["tool_name"] and r["query"] == p["tool_query"]:
#                 ordered.append(r); break

#     return {**state, "tool_results": ordered}

# # ========== Answer node ==========
# def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
#     llm = make_llm()

#     parts: List[str] = []
#     for item in state.get("tool_results", []):
#         tname = item.get("tool")
#         q = item.get("query")
#         out = item.get("output", {})

#         if tname == "weather":
#             if "error" in out:
#                 parts.append(f"[weather] ({q}) ERROR: {out['error']}")
#             else:
#                 parts.append(
#                     f"[weather] {out.get('location')} at {out.get('localtime')} → "
#                     f"{out.get('temperature_c')}°C, {out.get('condition')} "
#                     f"(feels {out.get('feels_like_c')}°C, humidity {out.get('humidity')}%, wind {out.get('wind_kph')} kph {out.get('wind_dir')})"
#                 )

#         elif tname == "mandi_price":
#             if "error" in out:
#                 parts.append(f"[mandi_price] ({q}) ERROR: {out['error']}")
#             else:
#                 lines = []
#                 for r in out.get("results", []):
#                     lines.append(
#                         f"{r.get('Arrival Date','—')} • {r.get('Market','—')} • Min {r.get('Min Price','—')} | "
#                         f"Avg {r.get('Avg Price','—')} | Max {r.get('Max Price','—')}"
#                     )
#                 head = f"[mandi_price] {out.get('state','?')}, {out.get('commodity','?')}:\n" if lines else "[mandi_price] No rows."
#                 parts.append(head + ("\n".join(lines) if lines else ""))

#         elif tname == "policy_pdf":
#             if "error" in out:
#                 parts.append(f"[policy_pdf] ({q}) ERROR: {out['error']}")
#             else:
#                 for r in out.get("results", []):
#                     src = (r.get("metadata", {}) or {}).get("source", "unknown")
#                     preview = (r.get("content") or "").strip().replace("\n", " ")
#                     if len(preview) > 300:
#                         preview = preview[:300] + "..."
#                     parts.append(f"[policy_pdf] {src}: {preview}")

#         elif tname == "web_search":
#             if "error" in out:
#                 parts.append(f"[web_search] ({q}) ERROR: {out['error']}")
#             else:
#                 for r in out.get("results", []):
#                     title = r.get("title","")
#                     url = r.get("url","")
#                     snip = (r.get("content","") or "").strip().replace("\n"," ")
#                     if len(snip) > 240: snip = snip[:240] + "..."
#                     parts.append(f"[web_search] {title} — {snip} (Source: {url})")

#         elif tname == "soil_nutrient":
#             if "error" in out:
#                 parts.append(f"[soil_nutrient] ({q}) ERROR: {out['error']}")
#             else:
#                 parts.append(f"[soil_nutrient] cycle={out.get('cycle')} state={out.get('state_name')} rows={len(out.get('results', []))}")

#         else:
#             parts.append(f"[{tname}] {json.dumps(out, ensure_ascii=False)[:400]}...")

#     context = "\n\n".join(parts)
#     user_q = state.get("english_input") or state.get("user_input") or ""
#     prompt = (
#         f"User question: {user_q}\n\n"
#         f"Available evidence (may be partial):\n{context}\n\n"
#         f"Compose a concise, actionable answer. If data is missing, say what is missing and suggest how to get it."
#     )

#     msgs = [SystemMessage(content="You are Krishi GPT, a farmer's helper which uses different tools attached to you and provide short solutions."),
#             HumanMessage(content=prompt)]
#     out = llm.invoke(msgs).content.strip()

#     return {**state, "final_answer": out}
