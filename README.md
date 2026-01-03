# 🤖 Textbook RAG Chatbot API

Single-file FastAPI chatbot for querying a robotics textbook using RAG.

## 📁 Files (Only 4!)

- **app.py** - Complete chatbot API (~400 lines, everything included)
- **requirements.txt** - Python dependencies
- **.env** - Your API keys (create this)
- **.gitignore** - Git ignore file

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create .env File
```bash
OPENAI_API_KEY=your_openai_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=textbook_chunks
```

### 3. Run the Server
```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### 1. Ask Question (RAG)
```
POST /chatbot/query
Content-Type: application/json

{
  "question": "What is Physical AI?",
  "top_k": 5
}
```

### 2. Ask About Selected Text
```
POST /chatbot/ask-selection
Content-Type: application/json

{
  "selected_text": "Your selected text here...",
  "question": "What does this mean?"
}
```

### 3. API Documentation
```
GET /docs - Interactive Swagger UI
GET /redoc - Alternative API docs
GET / - API info
```

## ✨ Features

- ✅ RAG-based question answering
- ✅ Text selection Q&A
- ✅ Source references with chapters
- ✅ OpenAI embeddings & GPT-4
- ✅ Qdrant vector database
- ✅ Single-file deployment
- ✅ Auto-generated API docs

**That's it! Everything in one file! 🎉**
