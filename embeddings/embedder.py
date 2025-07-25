import json
import os
import uuid
import tempfile
import asyncio
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment. Please check your .env file.")

def create_or_update_knowledge_base(bookmarks):
    # This sets up the event loop for Chroma if needed
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Set up the embeddings model
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    for bm in bookmarks:
        # Select a text field to use as the main content
        text = bm.get("full_text") or bm.get("text") or bm.get("title") or bm.get("content")
        if not text:
            continue
        # Collect extra information about the tweet
        metadata = {
            "likes": int(bm.get("likes", 0)),
            "retweets": int(bm.get("retweets", 0)),
            "views": int(bm.get("views", 0)),
            "tweet_url": bm.get("tweet_url", ""),
            "author": bm.get("author_name", ""),
            "author_handle": bm.get("author_handle", ""),
            "date": bm.get("tweet_date", ""),
            "content": text.strip()
        }
        # Split long texts so they fit in the embedding model
        chunks = splitter.split_documents([Document(page_content=text.strip(), metadata=metadata)])
        documents.extend(chunks)

    if not documents:
        raise ValueError("No valid content found in bookmarks.")

    # Give each user's bookmarks a unique name and temp folder
    collection_name = f"user_{uuid.uuid4().hex[:8]}"
    persist_dir = tempfile.mkdtemp()

    # Store all the embedded tweets in Chroma DB
    Chroma.from_documents(
        documents,
        embedding=gemini_embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return collection_name, gemini_embeddings, persist_dir, documents

def embed_bookmarks_from_file(uploaded_file):
    # Try to read the uploaded JSON file
    try:
        data = uploaded_file.read()
        bookmarks = json.loads(data)
    except Exception as e:
        print("Failed to read uploaded file:", e)
        return None, None, None, None

    # Check the input is a list of bookmarks
    if not bookmarks or not isinstance(bookmarks, list):
        print("Uploaded JSON must be a list of bookmark objects.")
        return None, None, None, None

    return create_or_update_knowledge_base(bookmarks)
