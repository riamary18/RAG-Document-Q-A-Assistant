import streamlit as st
import tempfile
import os
from src.pdf_loader import load_pdf, create_chunks
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.rag_chain import answer_question

st.set_page_config(page_title="InsightPDF", layout="wide")

st.title("📄 InsightPDF")
st.markdown("Upload a PDF and ask questions about its content.")
st.divider()

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Build RAG System
@st.cache_resource
def initialize_rag(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    try:
        pages = load_pdf(temp_path)
        chunks = create_chunks(pages)
        embeddings = get_embedding_model()
        vector_store = create_vector_store(chunks, embeddings)
        return vector_store, len(pages), len(chunks)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Application
if uploaded_file is not None:
    # Build knowledge base
    with st.spinner("Processing your document..."):
        vector_store, num_pages, num_chunks = initialize_rag(uploaded_file.getvalue())

    with st.sidebar:
        st.header("About")
        st.write("This application uses Retrieval-Augmented Generation (RAG) to answer questions from your uploaded PDF.")
        st.divider()

        st.header("Document")
        st.success("PDF loaded")
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Pages:** {num_pages}")
        st.divider()

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if not st.session_state.messages:
        st.markdown("### Ask your document anything")

    # Display conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if "sources" in message:
                    st.caption("Sources: " + ", ".join(f"Page {page}" for page in message["sources"]))

    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                answer, sources = answer_question(question, vector_store)
            st.markdown(answer)
            st.caption("Sources: " + ", ".join(f"Page {page}" for page in sources))

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# If no PDF was uploaded
else:
    st.markdown("### Get started")
    st.info("Upload a PDF above to build your document knowledge base and start chatting with it.")
    st.markdown(
        """
        **How it works**
        1. Upload a PDF
        2. The document is split into searchable chunks
        3. Relevant information is retrieved for each question
        4. An LLM generates an answer using the retrieved content
        """
    )