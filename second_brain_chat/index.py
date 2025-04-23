from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def index_chunks(chunks, metadata=None):
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma.from_documents(
        documents=[Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks],
        embedding=embedding_model,
    )
    return vectorstore

def search_chunks(query, store, top_k=3):
    if not query.strip():
        return []
    return store.similarity_search(query, k=top_k)
