# Second Brain Chat

LLM-powered chat interface for exploring your second-brain mindmaps (Freeplane-based).  
Powered by LangChain, ChromaDB, and OpenAI-compatible models.

## 📦 Setup

```bash
poetry install
cp .env.example .env  # edit your OpenAI base/key here
poetry run python second_brain_chat/chat_bot.py
