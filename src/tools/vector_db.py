import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import PDF_FOLDER, VECTOR_DB_DIR

# Lazy singleton
_policy_db = None

def get_policy_vector_db():
    global _policy_db
    if _policy_db is not None:
        return _policy_db

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(VECTOR_DB_DIR):
        _policy_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
        return _policy_db

    docs = []
    if os.path.isdir(PDF_FOLDER):
        for file in os.listdir(PDF_FOLDER):
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
                docs.extend(loader.load())

    _policy_db = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_DB_DIR)
    _policy_db.persist()
    return _policy_db
