# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langdetect import detect
# from src.io.ocr import ocr_image_to_text
# from src.io.pdf import extract_text_from_pdf
# from src.io.audio import transcribe_audio_file
# from src.llm.groq_client import make_llm
# from src.graph.build import build_graph
# from langchain.schema import SystemMessage, HumanMessage

# load_dotenv()

# def autodetect_lang(text: str) -> str:
#     try:
#         return detect(text)
#     except Exception:
#         return "en"

# def translate_to_english(text: str, source_lang: str):
#     if not text.strip():
#         return ""
#     llm = make_llm()
#     system = "Translate the following text to English, preserving meaning."
#     msgs = [SystemMessage(content=system), HumanMessage(content=text)]
#     return llm.invoke(msgs).content.strip()

# st.set_page_config(page_title="Krishi GPT", page_icon="🤖", layout="wide")
# st.title("Krishi GPT- the farmer's guide")

# workflow = build_graph()

# text_tab, image_tab, audio_tab, pdf_tab = st.tabs(["Text", "Image", "Audio", "PDF"])
# user_raw_text = ""

# with text_tab:
#     user_raw_text = st.text_area("Enter your question/content")

# with image_tab:
#     img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
#     if img_file:
#         from PIL import Image
#         img = Image.open(img_file)
#         extracted = ocr_image_to_text(img)
#         st.text_area("OCR text", value=extracted, height=150)
#         user_raw_text += "\n" + extracted

# with audio_tab:
#     aud_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
#     if aud_file:
#         transcription = transcribe_audio_file(aud_file.read())
#         st.text_area("Transcription", value=transcription, height=120)
#         user_raw_text += "\n" + transcription

# with pdf_tab:
#     pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
#     if pdf_file:
#         extracted_pdf = extract_text_from_pdf(pdf_file.read())
#         st.text_area("Extracted PDF text", value=extracted_pdf[:5000], height=180)
#         user_raw_text += "\n" + extracted_pdf

# lang = autodetect_lang(user_raw_text)
# english = translate_to_english(user_raw_text, lang)

# #st.text_area("English (LLM input)", value=english, height=180)

# if st.button("Run Agentic Pipeline"):
#     state = {"user_input": user_raw_text, "language": lang, "english_input": english}
#     result_state = workflow.invoke(state)
#     st.write(result_state.get("final_answer", ""))



import os
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
from src.io.ocr import ocr_image_to_text
from src.io.pdf import extract_text_from_pdf
from src.io.audio import transcribe_audio_file
from src.llm.groq_client import make_llm
from src.graph.build import build_graph
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

# ----- Style Enhancements -----
st.set_page_config(page_title="Krishi GPT", page_icon="🌾", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://unsplash.com/photos/soybean-field-rows-in-summer-17RKft4BeFM");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background-color: #f4f9f4;
}

textarea {
    border-radius: 12px !important;
    padding: 10px !important;
    font-size: 16px !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----- Title -----
st.markdown(
    "<h1 style='text-align: center; color: white;'>👨‍🌾 Krishi GPT – The Farmer's Guide</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #e0ffe0;'>Ask questions, upload images, audio, or PDFs and get farming guidance.</h4>", 
    unsafe_allow_html=True
)

# ----- Functions -----
def autodetect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def translate_to_english(text: str, source_lang: str):
    if not text.strip():
        return ""
    llm = make_llm()
    system = "Translate the following text to English, preserving meaning."
    msgs = [SystemMessage(content=system), HumanMessage(content=text)]
    return llm.invoke(msgs).content.strip()

workflow = build_graph()

# ----- Tabs with Icons -----
text_tab, image_tab, audio_tab, pdf_tab = st.tabs(["✍️ Text", "🖼️ Image", "🎙️ Audio", "📄 PDF"])
user_raw_text = ""

with text_tab:
    user_raw_text = st.text_area("Enter your question/content", placeholder="Type your farming question here...")

with image_tab:
    img_file = st.file_uploader("Upload farm-related image", type=["png", "jpg", "jpeg"])
    if img_file:
        from PIL import Image
        img = Image.open(img_file)
        extracted = ocr_image_to_text(img)
        st.text_area("Extracted OCR text", value=extracted, height=150)
        user_raw_text += "\n" + extracted

with audio_tab:
    aud_file = st.file_uploader("Upload audio message", type=["wav", "mp3", "m4a"])
    if aud_file:
        transcription = transcribe_audio_file(aud_file.read())
        st.text_area("Transcribed Audio", value=transcription, height=120)
        user_raw_text += "\n" + transcription

with pdf_tab:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        extracted_pdf = extract_text_from_pdf(pdf_file.read())
        st.text_area("Extracted PDF text", value=extracted_pdf[:5000], height=180)
        user_raw_text += "\n" + extracted_pdf

# ----- Processing -----
lang = autodetect_lang(user_raw_text)
english = translate_to_english(user_raw_text, lang)

# ----- Button -----
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🌱 Run Agentic Pipeline"):
    state = {"user_input": user_raw_text, "language": lang, "english_input": english}
    result_state = workflow.invoke(state)
    st.success(result_state.get("final_answer", ""))


