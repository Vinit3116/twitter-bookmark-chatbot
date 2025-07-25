import streamlit as st
from embeddings.embedder import embed_bookmarks_from_file
from chatbot.agent_langchain import build_agent

# Page setup
st.set_page_config(page_title="Twitter Bookmark Chatbot", layout="wide")
st.title("🧠 Twitter Bookmark Chatbot")
st.write("Chat with your saved Twitter bookmarks using Gemini & LangChain.")

# Sidebar: Description and sample questions
st.sidebar.title("About")
st.sidebar.write(
    "This app lets you search and browse your exported Twitter bookmarks.\n\n"
    "To use:\n"
    "1. Upload your `bookmarks.json` file\n"
    "2. Ask simple questions about your saved tweets."
)
st.sidebar.markdown("**Example questions:**")
st.sidebar.markdown("- Show my most recent bookmark")
st.sidebar.markdown("- List tweets about cricket")
st.sidebar.markdown("- Show tweets about Elon Musk")
st.sidebar.markdown("- Show my most liked tweet about Cricket")

# Upload bookmarks.json from Twitter
uploaded_file = st.file_uploader("📤 Upload your Twitter `bookmarks.json` file", type="json")

if uploaded_file:
    # Embed bookmarks to get a vector collection and docs
    with st.spinner("Processing your bookmarks..."):
        collection_name, embedding_function, persist_dir, all_docs = embed_bookmarks_from_file(uploaded_file)

    # Check for unsuccessful uploads
    if collection_name is None or all_docs is None:
        st.error("Could not process the uploaded file. Please upload a valid bookmarks JSON.")
        st.stop()
    else:
        st.success(f"{len(all_docs)} bookmarks loaded. You can now chat!")
        chain = build_agent(collection_name, embedding_function, persist_dir, all_docs)

        # Session chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # User submits a new question
        user_input = st.chat_input("Ask something about your bookmarks...")
        if user_input and chain:
            with st.spinner("Thinking..."):
                try:
                    # Get the agent's answer and save the exchange
                    result = chain.invoke({"question": user_input})
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.chat_history.append(("bot", result))
                except Exception as e:
                    st.error(f"Error: {e}")

        # Display chat history, alternating user/bot
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Bot:** {msg}")
else:
    st.info("Upload your Twitter bookmarks file to get started.")
