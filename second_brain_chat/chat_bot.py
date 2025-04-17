import os
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("LLM_MODEL", "Gemma-3-12b-it"),
    temperature=float(os.getenv("LLM_TEMP", "0.7")),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
)

try:
    chain = ConversationChain(llm=llm)
except Exception as e:
    print("Error calling LLM:", e)

if __name__ == "__main__":
    result = chain.invoke("Hello, world!")
    print(result["response"])
