# rag_pipeline.py
# Fully rewritten for modern LangChain (v0.2+)
# Uses LCEL (LangChain Expression Language) instead of deprecated chains

import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq

from embed_model import SentenceEmbedder

load_dotenv()


# ── PyTorch Embeddings wrapper ───────────────────────────────────────────────

class PyTorchEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceEmbedder()
        self.model.eval()
        print("✅ PyTorch embedding model loaded")

    def embed_query(self, text: str) -> list:
        return self.model.embed_query(text)

    def embed_documents(self, texts: list) -> list:
        return self.model.embed_documents(texts)


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Conversational RAG pipeline built with modern LangChain (LCEL).
    Remembers the last `memory_window` Q&A exchanges so follow-up
    questions like 'Tell me more' or 'Who wrote it?' resolve correctly.
    """

    def __init__(self, llm_model: str = "llama-3.1-8b-instant",
                 chunk_size: int = 500, chunk_overlap: int = 50,
                 memory_window: int = 5):

        self.embeddings   = PyTorchEmbeddings()
        self.vectorstore  = None
        self.chain        = None
        self.memory_window = memory_window
        self.chat_history  = []   # list of (HumanMessage, AIMessage) tuples

        self.llm = ChatGroq(
            model=llm_model,
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )

        # ── Prompt 1: condense follow-up question using history ──────────────
        self.condense_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the conversation history and a follow-up question, "
             "rewrite the follow-up as a standalone question that makes "
             "sense without the history. If it's already standalone, return it as-is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        # ── Prompt 2: answer using retrieved context ─────────────────────────
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Answer using ONLY the context below. "
             "If the answer is not in the context, say: "
             "'I don't have enough information in the document to answer this.'\n\n"
             "Context:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _format_docs(self, docs: list) -> str:
        return "\n\n".join(d.page_content for d in docs)

    def _get_windowed_history(self) -> list:
        """Return last memory_window exchanges as flat message list."""
        recent = self.chat_history[-self.memory_window:]
        messages = []
        for human_msg, ai_msg in recent:
            messages.append(human_msg)
            messages.append(ai_msg)
        return messages

    # ── Document loading ─────────────────────────────────────────────────────

    def load_document(self, file_path: str) -> list:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        print(f"📄 Loaded: {len(documents)} page(s)")
        return documents

    def load_text(self, text: str) -> list:
        return [Document(page_content=text, metadata={"source": "user_input"})]

    # ── Build vector store ───────────────────────────────────────────────────

    def build_vectorstore(self, documents: list) -> None:
        chunks = self.text_splitter.split_documents(documents)
        print(f"✂️  {len(chunks)} chunks created")

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.retriever   = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        print("🗄️  FAISS vector store ready")
        print("🔗 Conversational RAG chain ready!")

    # ── Ask ──────────────────────────────────────────────────────────────────

    def ask(self, question: str) -> dict:
        if not self.vectorstore:
            raise ValueError("No document loaded. Call build_vectorstore() first.")

        history = self._get_windowed_history()

        # Step 1 — condense follow-up into standalone question
        if history:
            condense_chain = self.condense_prompt | self.llm | StrOutputParser()
            standalone_q = condense_chain.invoke({
                "chat_history": history,
                "question": question
            })
        else:
            standalone_q = question

        # Step 2 — retrieve relevant chunks
        source_docs = self.retriever.invoke(standalone_q)
        context     = self._format_docs(source_docs)

        # Step 3 — generate answer
        answer_chain = self.answer_prompt | self.llm | StrOutputParser()
        answer = answer_chain.invoke({
            "chat_history": history,
            "context": context,
            "question": question
        })

        # Save to memory
        self.chat_history.append((HumanMessage(content=question),
                                   AIMessage(content=answer)))

        sources = [doc.page_content[:200] + "..." for doc in source_docs]
        return {"answer": answer, "sources": sources}

    def clear_memory(self) -> None:
        self.chat_history = []
        print("🧹 Conversation memory cleared")

    def get_chat_history(self) -> list:
        return self.chat_history

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_vectorstore(self, path: str = "vectorstore") -> None:
        self.vectorstore.save_local(path)
        print(f"💾 Saved to '{path}/'")

    def load_vectorstore(self, path: str = "vectorstore") -> None:
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        print(f"📂 Loaded from '{path}/'")


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  GROQ_API_KEY not found. Check your .env file.")
        exit(1)

    sample_text = """
    LangChain is a framework for building LLM-powered applications.
    It was created by Harrison Chase in 2022.

    RAG (Retrieval-Augmented Generation) retrieves relevant documents
    and provides them as context to an LLM to prevent hallucination.

    PyTorch is an open-source ML framework developed by Meta AI.
    Yann LeCun is the Chief AI Scientist at Meta.
    """

    with open("sample_doc.txt", "w") as f:
        f.write(sample_text)

    rag = RAGPipeline()
    rag.build_vectorstore(rag.load_document("sample_doc.txt"))

    for q in [
        "Who created LangChain?",
        "What year was it created?",
        "What is RAG?",
        "Why is it useful?",
    ]:
        r = rag.ask(q)
        print(f"Q: {q}\nA: {r['answer']}\n")