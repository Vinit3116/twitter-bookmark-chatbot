# streamlit_ui.py

import streamlit as st
from embeddings.embedder import embed_bookmarks_from_file
from chatbot.agent_langchain import build_agent

st.set_page_config(page_title="Twitter Bookmark Chatbot", layout="wide")
st.title("🧠 Twitter Bookmark Chatbot")
st.markdown("Chat with your saved Twitter bookmarks using Gemini + LangChain!")

st.sidebar.title("About")
st.sidebar.markdown("""
Built using:
- 🧠 LangChain
- 🤖 Gemini 2.5 Flash
- 🧱 ChromaDB
- 🐦 Twitter Bookmarks

Upload your exported Twitter bookmarks JSON and start chatting!
""")

uploaded_file = st.file_uploader("📤 Upload your Twitter bookmarks JSON file", type="json")

if uploaded_file:
    with st.spinner("🔍 Processing your bookmarks..."):
        collection_name, embedding_function, persist_dir = embed_bookmarks_from_file(uploaded_file)

        if collection_name is None:
            st.error("⚠️ Could not process the uploaded file. Please ensure it's a valid bookmarks JSON.")
        else:
            chain = build_agent(collection_name, embedding_function, persist_dir)
            st.success("✅ Bookmarks embedded. You can now chat!")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_input = st.chat_input("💬 Ask something about your bookmarks...")
            if user_input and chain:
                with st.spinner("🤔 Thinking..."):
                    result = chain.run(user_input)
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("bot", result))

            for role, msg in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(msg)
else:
    st.info("👆 Upload your Twitter bookmarks to get started.")
