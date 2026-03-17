# creates the index and retriever

from llama_index.core import VectorStoreIndex


def build_index(nodes, embed_model, storage_context):
    """
        Create vector index
    """
    index = VectorStoreIndex(
        nodes,
        embed_model = embed_model,
        storage_context = storage_context
    )
    return index

def get_retriever(index, top_k = 5):    
    """
        Creating retriever
    """
    retriever = index.as_retriever(
        similarity_top_k = top_k
    )
    return retriever