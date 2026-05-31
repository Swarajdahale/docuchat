# app.py
# Streamlit conversational UI for DocuChat
# Run: streamlit run app.py

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuChat",
    page_icon="📚",
    layout="wide"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "rag"          not in st.session_state: st.session_state.rag          = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "doc_name"     not in st.session_state: st.session_state.doc_name     = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 DocuChat")
    st.caption("Conversational Q&A over your documents")
    st.divider()

    # API Key
    st.subheader("⚙️ Groq API Key")
    env_key      = os.getenv("GROQ_API_KEY", "")
    key_from_env = bool(env_key and not env_key.startswith("gsk_your"))

    if key_from_env:
        st.success("✅ Key loaded from .env")
        api_key = env_key
    else:
        api_key = st.text_input("Enter Groq API Key", type="password",
                                help="Free key at console.groq.com")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

    # Model
    st.subheader("🤖 Model")
    model_choice = st.selectbox("Groq Model", [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama4-scout-17b-16e-instruct",
        "gemma2-9b-it",
        "qwen/qwen3-32b",
    ])

    st.divider()

    # Document upload
    st.subheader("📤 Load Document")
    input_method = st.radio("Input", ["Upload File", "Paste Text"],
                            label_visibility="collapsed")

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose .txt or .pdf", type=["txt", "pdf"])
        if uploaded_file and api_key:
            if st.button("⚡ Process Document", type="primary", use_container_width=True):
                with st.spinner("Building RAG pipeline..."):
                    suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                        f.write(uploaded_file.read())
                        tmp_path = f.name
                    rag = RAGPipeline(llm_model=model_choice)
                    docs = rag.load_document(tmp_path)
                    rag.build_vectorstore(docs)
                    os.unlink(tmp_path)
                    st.session_state.rag          = rag
                    st.session_state.chat_history = []
                    st.session_state.doc_name     = uploaded_file.name
                st.success(f"✅ Ready! Ask about **{uploaded_file.name}**")
        elif not api_key:
            st.warning("Add your Groq API key first.")

    else:
        pasted_text = st.text_area("Paste text here", height=150,
                                   label_visibility="collapsed",
                                   placeholder="Paste document content here...")
        if pasted_text and api_key:
            if st.button("⚡ Process Text", type="primary", use_container_width=True):
                with st.spinner("Building RAG pipeline..."):
                    rag = RAGPipeline(llm_model=model_choice)
                    docs = rag.load_text(pasted_text)
                    rag.build_vectorstore(docs)
                    st.session_state.rag          = rag
                    st.session_state.chat_history = []
                    st.session_state.doc_name     = "Pasted Text"
                st.success("✅ Ready! Ask anything about your text.")
        elif not api_key:
            st.warning("Add your Groq API key first.")

    st.divider()

    # Conversation controls
    if st.session_state.rag:
        st.subheader("💬 Conversation")
        exchanges = len(st.session_state.chat_history) // 2
        st.caption(f"{exchanges} exchange(s) in memory")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.rag.clear_memory()
            st.rerun()

    st.divider()
    with st.expander("🏗️ How it works"):
        st.markdown("""
1. **Split** — Doc → 500-char chunks  
2. **Embed** — PyTorch model → vectors  
3. **Store** — FAISS vector DB  
4. **Condense** — follow-up Q + history → standalone Q  
5. **Retrieve** — top 4 relevant chunks  
6. **Generate** — Groq LLM answers  

Memory keeps last **5 exchanges** so follow-ups
like *"Tell me more"* work correctly.
        """)

# ── Main chat area ────────────────────────────────────────────────────────────
if st.session_state.doc_name:
    st.header(f"💬 {st.session_state.doc_name}")
else:
    st.header("💬 DocuChat")
    st.info("👈 Upload a document and add your Groq API key to get started.")

# Render history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Retrieved chunks", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(src)

# Chat input
if st.session_state.rag:
    if question := st.chat_input("Ask anything about your document..."):
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result  = st.session_state.rag.ask(question)
            answer  = result["answer"]
            sources = result["sources"]
            st.markdown(answer)
            if sources:
                with st.expander("📎 Retrieved chunks", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(src)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })