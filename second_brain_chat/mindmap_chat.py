#!/usr/bin/env python3
import argparse, os
from langchain.schema import Document
from second_brain_chat.freeplane_parser import parse_mm, chunk_node
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_index(mm_path: str, db_dir: str = "chroma_db"):
    root = parse_mm(mm_path)
    chunks = chunk_node(root, max_tokens=500)
    os.makedirs(db_dir, exist_ok=True)

    # initialize the HuggingFaceEmbeddings wrapper
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # wrap each chunk in a langchain.schema.Document
    docs = [
        Document(page_content=chunk, metadata={"source": os.path.basename(mm_path)})
        for chunk in chunks
    ]

    # create & persist a Chroma vector store
    store = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        persist_directory=db_dir,
    )
    return store

def load_index(db_dir: str = "chroma_db"):
    # same embedder used for querying
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        embedding=embedder,
        persist_directory=db_dir,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", help="Path to .mm mindmap")
    p.add_argument("--reindex", action="store_true", help="Rebuild index from scratch")
    args = p.parse_args()

    db_dir = f"chroma_db_{os.path.basename(args.file)}"
    if args.reindex or not os.path.isdir(db_dir):
        store = build_index(args.file, db_dir)
    else:
        store = load_index(db_dir)

    print("Index ready. Type your question (or 'quit'):")
    while True:
        q = input(">> ").strip()
        if q.lower() in ("quit", "exit"):
            break
        results = store.similarity_search(q, k=5)
        for doc in results:
            print(f"---\n{doc.page_content}\n")

if __name__ == "__main__":
    main()
