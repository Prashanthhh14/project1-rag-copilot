# dashboard.py — Streamlit Visual Dashboard
# A beautiful chat interface for our RAG system

import streamlit as st
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our RAG pipeline
from rag_pipeline import retrieve_chunks, build_prompt, generate_answer

# Import ChromaDB
import chromadb

# ─────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────

# This must be the FIRST streamlit command
# Sets up the browser tab title and layout
st.set_page_config(
    page_title="LexTech Support Copilot",
    page_icon="🤖",
    layout="wide",
    # wide = use full browser width
)


# ─────────────────────────────────────────────────
# LOAD RESOURCES (once, not on every question)
# ─────────────────────────────────────────────────

# @st.cache_resource means:
# "run this function ONCE and remember the result"
# Without this, the model would reload every time
# the user asks a question — very slow!
@st.cache_resource
def load_resources():
    """
    Loads embedding model and ChromaDB once.
    Cached so it only runs on first page load.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(
        path="embeddings/chroma_db"
    )
    collection = chroma_client.get_collection(
        "lextech_knowledge_base"
    )
    return model, collection

# Load everything
model, collection = load_resources()


# ─────────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────────

# st.markdown with unsafe_allow_html lets us add CSS
# CSS = the language that styles web pages
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
    }
    .stChatMessage {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">🤖 LexTech Support Copilot</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Ask anything about LexTech — '
    'powered by RAG + Llama AI</div>',
    unsafe_allow_html=True
)

st.divider()  # horizontal line


# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────

# st.sidebar = the left panel
with st.sidebar:
    st.header("⚙️ Settings")

    # Slider to control how many chunks to retrieve
    n_results = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help="More chunks = more context but slower response"
    )

    st.divider()

    # Database stats
    st.header("📊 Knowledge Base")
    total_chunks = collection.count()
    st.metric("Total Chunks", total_chunks)
    st.metric("Documents", 5)
    st.metric("AI Model", "Llama 3.2")
    st.metric("Embeddings", "MiniLM-L6")

    st.divider()

    # Sample questions for quick testing
    st.header("💡 Try These Questions")
    sample_questions = [
        "How do I reset my password?",
        "What is the refund policy?",
        "How do I invite team members?",
        "What file types can I upload?",
        "How do I enable 2FA?",
        "What are the pricing plans?",
    ]

    for question in sample_questions:
        if st.button(question, use_container_width=True):
            # When button is clicked, set it as the current question
            st.session_state.clicked_question = question

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        # st.rerun() refreshes the page


# ─────────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────────

# session_state stores data between interactions
# Without it, everything resets when user types
# Think of it like the app's short-term memory

# Initialize chat history if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Add a welcome message from the bot
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm the LexTech Support Copilot. "
                   "I can answer questions about LexTech using "
                   "our official documentation. What would you "
                   "like to know?",
        "sources": [],
        "response_time": None
    })

# Display all previous messages
for message in st.session_state.messages:

    # st.chat_message creates a chat bubble
    # "user" = right side, "assistant" = left side
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show sources if this was an AI response
        if message.get("sources"):
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>[{i+1}] {source["source"]}'
                        f'</strong> — {source["heading"]}<br>'
                        f'<small>Relevance score: '
                        f'{source["score"]:.4f}</small><br>'
                        f'<small>{source["preview"]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # Show response time if available
        if message.get("response_time"):
            st.caption(
                f"⏱️ Response time: "
                f"{message['response_time']:.1f} seconds"
            )


# ─────────────────────────────────────────────────
# HANDLE INPUT
# ─────────────────────────────────────────────────

# Check if a sample question button was clicked
if "clicked_question" in st.session_state:
    user_input = st.session_state.clicked_question
    del st.session_state.clicked_question
    # del = delete this from session_state
    # so it doesn't keep triggering
else:
    user_input = None

# Chat input box at the bottom of the page
chat_input = st.chat_input(
    "Ask a question about LexTech..."
)

# Use either the chat input or clicked button
question = chat_input or user_input

# Process the question if we have one
if question:

    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": [],
        "response_time": None
    })

    # Display user message immediately
    with st.chat_message("user"):
        st.write(question)

    # Generate AI response
    with st.chat_message("assistant"):

        # Show a spinner while generating
        with st.spinner("🔍 Searching knowledge base and "
                        "generating answer..."):

            start_time = time.time()

            # Step 1: Retrieve chunks
            chunks = retrieve_chunks(question, n_results)

            # Step 2: Build prompt
            prompt = build_prompt(question, chunks)

            # Step 3: Generate answer
            answer = generate_answer(prompt)

            response_time = time.time() - start_time

        # Display the answer
        st.write(answer)

        # Format sources for display
        sources = []
        for chunk in chunks:
            sources.append({
                "source": chunk["source"],
                "heading": chunk["heading"],
                "score": chunk["score"],
                "preview": chunk["content"][:150]
            })

        # Show sources in expandable section
        with st.expander("📚 View Sources"):
            for i, source in enumerate(sources):
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>[{i+1}] {source["source"]}'
                    f'</strong> — {source["heading"]}<br>'
                    f'<small>Relevance score: '
                    f'{source["score"]:.4f}</small><br>'
                    f'<small>{source["preview"]}...</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Show response time
        st.caption(f"⏱️ Response time: {response_time:.1f} seconds")

        # Save assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "response_time": response_time
        })


# ─────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────

st.divider()
col1, col2, col3 = st.columns(3)
# st.columns splits the page into 3 equal columns

with col1:
    st.markdown("**🤖 Model:** Llama 3.2 (Local)")
with col2:
    st.markdown("**🔍 Search:** ChromaDB + MiniLM")
with col3:
    st.markdown("**📚 Docs:** 5 LexTech documents")