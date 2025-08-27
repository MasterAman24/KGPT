import requests
from typing import Any, Dict
from .config import WEATHERAPI_KEY

def weather_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    q = state.get("tool_query")
    if not q:
        return {**state, "tool_result": {"error": "Empty query"}}
    if not WEATHERAPI_KEY:
        return {**state, "tool_result": {"error": "Missing WEATHERAPI_KEY"}}

    city = str(q).strip()
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
