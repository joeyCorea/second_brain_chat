import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
import tiktoken

# Token counter using OpenAI encoding (works for GPT-like models)
def count_tokens(text, encoding_name="cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

# Set up local embedding model and LangChain-compatible wrapper
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_fn = lambda texts: embedding_model.encode(texts, normalize_embeddings=True)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embedding, persist_directory=None)

# Memory stub — stores last N interactions
chat_memory = []

# Set up environment variables
load_dotenv()

# Skip LM Studio validation in CI (GitHub Actions, etc.)
if os.getenv("CI") == "true":
    print("⚠️ Skipping LM Studio server validation in CI environment.")
else:
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
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

def run_chat_turn(user_input: str, vectorstore, llm_chain) -> str:
    # Add the user's input to the vector store
    vectorstore.add_texts([user_input])    
    # Retrieve relevant context from the vectorstore
    context = "\n".join([r.page_content for r in vectorstore.similarity_search(user_input, k=3)])
    # Use the passed-in llm_chain to generate a response
    response = llm_chain.invoke({"input": user_input, "context": context})
    # Return the content of the response (i.e., assistant's reply)
    return response.content

# Main loop for interactive chatting
def start_chat():
    print("Start chatting with your local-memory assistant. Type 'exit' to quit.")
    
    # Construct the prompt and LLM chain outside of the function
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that uses memory snippets to respond."),
        ("human", "{input}\n\nRelevant memory:\n{context}")
    ])
    
    llm_chain = prompt | llm  # Construct the llm_chain for this session
    
    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() == "exit":
            break
        
        # Call the chat function
        assistant_response = run_chat_turn(user_input, vectorstore, llm_chain)
        
        # Count and display the number of tokens in the assistant's response
        response_tokens = count_tokens(assistant_response)
        print(f"[Response token count: {response_tokens}]")
        
        # Display assistant's response
        print(f"\nAssistant:\n{assistant_response}")
        
        # Add interaction to chat memory log
        chat_memory.append((user_input, assistant_response))

if __name__ == "__main__":
    start_chat()
