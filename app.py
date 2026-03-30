# Streamlit Cloud SQLite patch for ChromaDB

# Chromadb requires a minimum SQLite3 version of 3.35. Streamlit Cloud's default environment 
# uses an older version of SQLite3, causing apps to crash on load. This patch safely swaps the
#  SQLite version during runtime on the cloud.

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st

from src.ingest import load_documents, chunk_documents
from src.embeddings import get_embedding_model
from src.vectorstore import get_vector_store
from src.retriever import build_index, get_retriever
from src.rag_pipeline import get_llm, generate_answer


@st.cache_resource
def initialize_rag():

    documents = load_documents()
    nodes = chunk_documents(documents)

    embed_model = get_embedding_model()

    storage_context = get_vector_store()

    index = build_index(nodes, embed_model, storage_context)

    retriever = get_retriever(index)

    llm = get_llm()

    return retriever, llm


retriever, llm = initialize_rag()

st.title("Brototype Guidelines Assistant")

st.write("Shoot me questions to help you avoid warning mails & suspensions: ")

question = st.text_input("Ask a question:")

if question:

    answer = generate_answer(question, retriever, llm)

    st.write("### Answer")
    st.write(answer)