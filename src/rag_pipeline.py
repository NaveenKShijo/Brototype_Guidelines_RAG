
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_API_KEY, LLM_MODEL

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model = LLM_MODEL,
        google_api_key = GEMINI_API_KEY,
        temperature = 0
    )
    return llm

from langchain_core.messages import SystemMessage, HumanMessage

def build_messages(context, question):
    system_prompt = (
        "You are a helpful assistant. Answer only using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        "If the answer is not in the context, say: \"I could not find the answer in the documents\""
    )
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

def generate_answer(question, retriever, llm):
    nodes = retriever.retrieve(question)
    # in LlamaIndex, .get_content() is standard for retrieving text from NodeWithScore
    context = "\n\n".join([node.get_content() for node in nodes])
    messages = build_messages(context, question)
    response = llm.invoke(messages)
    return response.content