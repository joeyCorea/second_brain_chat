import os
import requests
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# Validate LM Studio availability
try:
    requests.get(os.getenv("OPENAI_API_BASE"), timeout=2)
except Exception:
    print("⚠️ LM Studio server not reachable at", os.getenv("OPENAI_API_BASE"))
    exit(1)

# Set up the LLM
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY", "not-needed-for-local"),
    model_name=os.getenv("LLM_MODEL", "Gemma-3-12b-it"),
    temperature=float(os.getenv("LLM_TEMP", "0.7")),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant with access to the user's notes."),
    ("human", "{input}")
])

# Use prompt + model as a LangChain Runnable
chain = prompt | llm

# Chat loop
while True:
    query = input("\nYou> ")
    if query.strip().lower() in {"exit", "quit"}:
        break
    try:
        result = chain.invoke({"input": query})
        print("\nAssistant>\n", result.content)
    except Exception as e:
        print("❌ Error calling LLM:", e)
