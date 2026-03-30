from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_documents():
    """
        Load data sources and convert them into documents using an explicit PDF reader
    """

    # documents = SimpleDirectoryReader(DATA_DIR).load_data()
    # it failed as it downgraded to return raw pdf structure instead of readable text.
    # So we explicitly use PDFReader.

    loader = SimpleDirectoryReader(
        DATA_DIR,
        file_extractor={".pdf": PDFReader()}
    )
    documents = loader.load_data()
    return documents

def chunk_documents(documents):
    """
        Split documents into chunks
    """
    splitter = SentenceSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes