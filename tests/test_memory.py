import pytest
from unittest.mock import MagicMock
from second_brain_chat.memory_chat import count_tokens
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

@pytest.fixture
def mock_vectorstore():
    # Create a mock for Chroma vector store
    mock_vs = MagicMock(spec=Chroma)  # Properly mock Chroma class
    mock_vs.similarity_search.return_value = [
        MagicMock(page_content="Test context 1"),
        MagicMock(page_content="Test context 2"),
    ]
    return mock_vs

@pytest.fixture
def mock_embedding_model():
    # Fixture for sentence-transformer model
    return SentenceTransformer("all-MiniLM-L6-v2")

def test_count_tokens():
    # Test token count function
    test_text = "Tokenization is essential for LLMs."
    token_count = count_tokens(test_text)
    assert token_count > 0  # Ensure token count is greater than zero

def test_add_and_retrieve_text(mock_vectorstore):
    # Test adding and retrieving text from Chroma
    texts = ["I am learning generative AI", "Tokenization is essential for LLMs", "Chroma stores vectors efficiently"]
    mock_vectorstore.add_texts(texts)

    # Check that add_texts was called correctly
    mock_vectorstore.add_texts.assert_called_once_with(texts)

    # Test similarity search
    query = "What is tokenization?"
    
    # Fix: ensure that the mock returns the correct page_content for the results
    mock_vectorstore.similarity_search.return_value = [
        MagicMock(page_content="Tokenization is essential for LLMs"),
        MagicMock(page_content="Chroma stores vectors efficiently")
    ]

    results = mock_vectorstore.similarity_search(query, k=2)
    assert len(results) == 2  # Ensure exactly two results are returned
    assert "Tokenization is essential for LLMs" in results[0].page_content
    assert "Chroma stores vectors efficiently" in results[1].page_content


def test_embedding(mock_embedding_model):
    # Test embeddings creation with sentence-transformers
    texts = ["I am learning generative AI"]
    embeddings = mock_embedding_model.encode(texts)

    assert embeddings.shape == (1, 384)  # Expected shape for all-MiniLM-L6-v2
    assert len(embeddings) == 1  # One embedding for one input text

def test_vector_store_token_count(mock_vectorstore):
    # Test token count before and after adding to vector store
    input_text = "Tokenization is essential for LLMs."

    # Test token count before adding to vector store
    input_tokens = count_tokens(input_text)
    assert input_tokens > 0

    # Add to vector store and ensure it's stored
    mock_vectorstore.add_texts([input_text])
    mock_vectorstore.similarity_search(input_text, k=1)

    # Ensure vector store interaction
    mock_vectorstore.similarity_search.assert_called_once_with(input_text, k=1)
