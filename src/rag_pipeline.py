
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_API_KEY, LLM_MODEL

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model = LLM_MODEL,
        google_api_key = GEMINI_API_KEY,
        temperature = 0
    )
    return llm

def build_prompt(context, question):
    prompt = f"""
        You are a helpful assistant. Answer only using the provided context.

        Context:
        {context}

        Question:
        {question}

        If the answer is not in the context, say: "I could not find the answer in the documents"
    """
    return prompt

def generate_answer(question, retriever, llm):
    nodes = retriever.retrieve(question)
    context = "\n\n".join([node.text for node in nodes])
    prompt = build_prompt(context, question)
    response = llm.invoke(prompt)
    return response.content