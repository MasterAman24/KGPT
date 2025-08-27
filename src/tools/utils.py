import re
from typing import Tuple
from .config import INDIA_STATES_UTS

def extract_city_for_weather(text: str) -> str:
    text_l = text.lower()
    m = re.search(r"(?:in|at)\s+([a-zA-Z .'-]+)", text_l)
    if m:
        city = m.group(1).strip(" .'-")
        city = re.split(r"\b(today|now|tomorrow|this week|forecast)\b", city)[0].strip()
        return city.title()
    tokens = re.findall(r"[A-Z][a-zA-Z'-]+", text)
    if tokens:
        return tokens[-1]
    return text.strip()

def extract_mandi_state_commodity(text: str) -> Tuple[str, str]:
    t = text.lower()
    found_state = None
    for st in INDIA_STATES_UTS:
        if st in t:
            found_state = st
            break
    m = re.search(r"(?:price|prices|rate|rates)\s+(?:of|for)\s+([a-zA-Z /'-]+)", t)
    commodity = None
    if m:
        commodity = m.group(1).strip(" .'-")
    if not commodity and found_state:
        before = t.split(found_state)[0]
        last = re.findall(r"[a-z]+", before)
        if last:
            commodity = last[-1]
    if not found_state:
        if "delhi" in t or "ncr" in t:
            found_state = "nct of delhi"
    if not commodity:
        commodity = "wheat"
    if not found_state:
        found_state = "rajasthan"
    return (found_state.title(), commodity.title())
