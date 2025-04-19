from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
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

print("Start chatting with your local-memory assistant. Type 'exit' to quit.")
while True:
    user_input = input("\nUser: ")
    if user_input.strip().lower() == "exit":
        break
    user_tokens = count_tokens(user_input)
    print(f"[User input token count: {user_tokens}]")
    # Save to memory vectorstore
    vectorstore.add_texts([user_input])
    # Retrieve context from vectorstore (top-3 most similar chunks)
    results = vectorstore.similarity_search(user_input, k=3)
    retrieved_context = "\n".join([r.page_content for r in results])
    context_tokens = count_tokens(retrieved_context)
    print(f"[Retrieved context token count: {context_tokens}]")
    # Generate fake assistant response (stub/mock for now)
    assistant_response = f"(Mock response based on context)\n{retrieved_context[:300]}..."
    response_tokens = count_tokens(assistant_response)
    print(f"[Response token count: {response_tokens}]")
    print(f"\nAssistant:\n{assistant_response}")
    # Add interaction to chat memory log
    chat_memory.append((user_input, assistant_response))
