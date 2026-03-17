from src.ingest import load_documents, chunk_documents
from src.embeddings import get_embedding_model
from src.rag_pipeline import get_llm, build_prompt, generate_answer
from src.retriever import build_index, get_retriever
from src.vectorstore import get_vector_store


def initialize_rag():
    print("Loading documents")
    documents = load_documents()

    print("Chunking documents")
    nodes = chunk_documents(documents)

    print("Loading embedding model")
    embed_model = get_embedding_model()

    print("Connecting vector database")
    storage_context = get_vector_store()

    print("Building index")
    index = build_index(nodes, embed_model, storage_context)

    retriever = get_retriever(index)
    llm = get_llm()

    return retriever, llm


def run():
    retriever, llm = initialize_rag()
    print("\nRAG system Ready!\n")
    while True:
        question = input("Chat with me: (To quit, type: 'exit')")
        if question.lower().strip() == 'exit':
            break
        answer = generate_answer(question, retriever, llm)
        print("\nAnswer:\n", answer)
        print("\n"+"-"*50 + "\n")

if __name__ == '__main__':
    run()     

# Now run() only fires when you run this file directly. When someone imports main.py, run() stays silent.