import os

# ====== Vector DB for PDFs ======
PDF_FOLDER = os.getenv("POLICY_PDF_FOLDER", "Major Schemes")
VECTOR_DB_DIR = os.getenv("POLICY_VECTOR_DB_DIR", "vector_db_policy")

# ====== Weather API ======
WEATHERAPI_KEY = os.getenv("OPENWEATHER_API_KEY", "75c43d92e1f8407590b205917251108")

# ====== India states / UTs (for mandi parsing & soil nutrient)
INDIA_STATES_UTS = {
    "andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh","goa","gujarat","haryana","himachal pradesh",
    "jharkhand","karnataka","kerala","madhya pradesh","maharashtra","manipur","meghalaya","mizoram","nagaland",
    "odisha","punjab","rajasthan","sikkim","tamil nadu","telangana","tripura","uttar pradesh","uttarakhand","west bengal",
    "andaman and nicobar islands","chandigarh","dadra and nagar haveli and daman and diu","delhi","lakshadweep",
    "puducherry","jammu and kashmir","ladakh","nct of delhi","ncr"
}
