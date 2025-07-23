# embeddings/embedder.py

import json
import os
import uuid
import tempfile
import asyncio
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scraper.twitter_scraper import load_bookmarks

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("🚨 GOOGLE_API_KEY not found in environment. Please check your .env file.")

def create_or_update_knowledge_base(bookmarks):
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    texts = []
    for bm in bookmarks:
        text = bm.get("full_text") or bm.get("text") or bm.get("title") or bm.get("content")
        if text:
            texts.append(text.strip())

    if not texts:
        raise ValueError("No valid bookmark text found to embed.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    documents = [Document(page_content=t) for t in texts]
    split_docs = splitter.split_documents(documents)
    split_texts = [doc.page_content for doc in split_docs]

    embeddings = gemini_embeddings.embed_documents(split_texts)
    if not embeddings:
        raise ValueError("Gemini returned empty embeddings.")

    collection_name = f"user_{uuid.uuid4().hex[:8]}"
    persist_dir = tempfile.mkdtemp()

    Chroma.from_texts(
        texts=split_texts,
        embedding=gemini_embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    return collection_name, gemini_embeddings, persist_dir

def embed_bookmarks_from_file(uploaded_file):
    try:
        data = uploaded_file.read()
        bookmarks = json.loads(data)
    except Exception as e:
        print("❌ Failed to read uploaded file:", e)
        return None, None, None

    if not bookmarks or not isinstance(bookmarks, list):
        print("❌ Uploaded JSON must be a list of bookmark objects.")
        return None, None, None

    return create_or_update_knowledge_base(bookmarks)
