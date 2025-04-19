import pytest
from unittest.mock import MagicMock
from second_brain_chat.memory_chat import run_chat_turn  # Corrected import path

@pytest.fixture
def mock_vectorstore():
    # Mocking the Chroma vector store
    vs = MagicMock()
    vs.similarity_search.return_value = [
        MagicMock(page_content="Test context 1"),
        MagicMock(page_content="Test context 2"),
    ]
    return vs

@pytest.fixture
def mock_llm_chain():
    # Mocking the LLM chain
    llm_chain = MagicMock()
    llm_chain.invoke.return_value.content = "Test response"
    return llm_chain

def test_run_chat_turn(mock_vectorstore, mock_llm_chain):
    # Given
    user_input = "What is the capital of France?"

    # When
    response = run_chat_turn(user_input, mock_vectorstore, mock_llm_chain)

    # Then
    assert "Test response" in response
    mock_vectorstore.add_texts.assert_called_once_with([user_input])
    mock_vectorstore.similarity_search.assert_called_once_with(user_input, k=3)
    mock_llm_chain.invoke.assert_called_once()

