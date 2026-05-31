# 📚 DocuChat — Conversational AI Document Q&A

> Upload any PDF or text document and have a full conversation with it using RAG, LangChain, and PyTorch — powered by Groq's free LLM API.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Groq](https://img.shields.io/badge/Groq-Free%20API-orange)

---

## 🎯 What it does

DocuChat lets you upload a PDF or paste any text, then ask questions about it in a natural conversational way. Follow-up questions like *"Tell me more"* or *"Who wrote it?"* work correctly because the app remembers your conversation history.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
[Condense Step] ── rewrites follow-up using chat history → standalone question
     │
     ▼
[PyTorch Embedder] ── converts question into a 256-dim vector
     │
     ▼
[FAISS Vector DB] ── finds top-4 most similar document chunks
     │
     ▼
[Groq LLM] ── reads chunks + question → generates grounded answer
     │
     ▼
Answer + Source Chunks shown in UI
```

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **PyTorch** | Custom `SentenceEmbedder` model with `nn.Linear` projection layer |
| **HuggingFace Transformers** | Base transformer (`all-MiniLM-L6-v2`) inside PyTorch model |
| **LangChain (LCEL)** | RAG pipeline, document loading, prompt chaining |
| **FAISS** | Vector database for fast similarity search |
| **Groq API** | Free, fast LLM inference (Llama 3, Gemma, Qwen) |
| **Streamlit** | Conversational web UI |
| **python-dotenv** | Secure API key management via `.env` file |

---

## 📁 Project Structure

```
docuchat/
├── .env                  ← your real API key (never commit this)
├── .env.example          ← safe template to share
├── .gitignore            ← keeps .env and venv out of GitHub
├── embed_model.py        ← PyTorch SentenceEmbedder (nn.Module)
├── rag_pipeline.py       ← LangChain LCEL RAG + conversational memory
├── app.py                ← Streamlit chat UI
├── requirements.txt      ← all pip dependencies
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/docuchat.git
cd docuchat
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your free Groq API key
```bash
# Get your free key at https://console.groq.com (no credit card needed)
cp .env.example .env
# Open .env and replace the placeholder with your real key
```

`.env` file:
```
GROQ_API_KEY=gsk_your_real_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy for Free (Streamlit Community Cloud)

1. Push your code to GitHub *(make sure `.env` is NOT pushed)*
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file as `app.py`
4. Click **Advanced settings** → **Secrets** → add:
```
GROQ_API_KEY = "gsk_your_real_key_here"
```
5. Click **Deploy** → get your live public URL in ~3 minutes ✅

---

## 🤖 Available Groq Models (all free)

| Model | Best For |
|---|---|
| `llama-3.1-8b-instant` | Fast responses, everyday Q&A (default) |
| `llama-3.3-70b-versatile` | Smarter, more detailed answers |
| `llama4-scout-17b-16e-instruct` | Latest Llama 4, great balance |
| `gemma2-9b-it` | Google's Gemma, good instruction following |
| `qwen/qwen3-32b` | Strong reasoning |

---

## 🎤 Interview Q&A Guide

### "Walk me through your project"
> "I built DocuChat, an AI system where you upload a document and chat with it conversationally. The core idea is RAG — instead of relying on the LLM's general knowledge, I first retrieve the most relevant parts of the document and pass them as context. This keeps answers grounded and prevents hallucination."

### "What is RAG and why did you use it?"
> "RAG stands for Retrieval-Augmented Generation. The problem with asking an LLM about a specific document is it might use its training data and hallucinate. RAG solves this by converting the document into vector embeddings, storing them in a database, and at query time retrieving the most relevant chunks to include in the prompt. The LLM then answers based only on those chunks."

### "Where does PyTorch fit in?"
> "I built a custom SentenceEmbedder class inheriting from nn.Module. It wraps a HuggingFace transformer and adds a trainable nn.Linear projection layer that maps embeddings to 256 dimensions. Key concepts used: nn.Module, nn.Sequential, nn.LayerNorm, torch.no_grad() for inference, and L2 normalization for cosine similarity search."

### "How does conversational memory work?"
> "I store each Q&A exchange as a (HumanMessage, AIMessage) pair in a list. Before retrieval, a condense step rewrites the new question using the last 5 exchanges into a fully standalone question. So if you ask 'Who wrote it?' after asking about a paper, the system rewrites it to 'Who wrote [the paper title]?' before searching — that's what makes follow-ups work."

### "Why did you choose Groq?"
> "Groq offers a completely free API tier with no credit card required, and their LPU hardware makes inference significantly faster than GPU-based providers. It was also easy to integrate with LangChain via langchain-groq. For a project focused on learning RAG and LangChain, it removed the cost barrier entirely."

### "What challenges did you face?"
> "LangChain released v0.2 which deprecated the older ConversationalRetrievalChain and moved modules into separate packages like langchain-core and langchain-text-splitters. I had to migrate the entire pipeline to LCEL — LangChain Expression Language — which is the modern approach. It actually gave me more control over each step of the pipeline."

### "What are the limitations?"
> "Fixed chunk size — adaptive semantic chunking would be better. Basic similarity retrieval — adding a reranker would improve quality. The PyTorch projection layer is not fine-tuned for any specific domain. And FAISS is in-memory, so for production I'd replace it with a managed vector DB like Pinecone."

### "How would you scale this?"
> "Replace FAISS with Pinecone or Weaviate for persistent managed storage, add async processing for large documents, cache embeddings to avoid recomputing them, and add streaming for LLM responses so the UI feels faster."

---

## 📄 License

MIT License — free to use, modify, and distribute.