import os
import json
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in .env file. Please set it in your .env file.")
    exit(1)

BOOKMARKS_FILE = 'scraper/bookmarks.json'
CHROMA_COLLECTION_NAME = 'twitter_bookmarks'
CHROMA_DB_PATH = './chroma_db'


# --- Gemini Embedding Function ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "models/embedding-001"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        print(f"✅ Gemini embedding model '{self.model_name}' configured.")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            try:
                response = genai.embed_content(
                    model=self.model_name,
                    content=text,  # ✅ pass single string
                    task_type="retrieval_document",
                    title="Tweet Bookmark"
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                print(f"❌ Error embedding text: {text[:50]}... → {e}")
                embeddings.append([0.0] * 768)
        return embeddings


# --- Load bookmarks ---
def load_bookmarks(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Bookmarks file not found: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            bookmarks = json.load(f)
        print(f"✅ Loaded {len(bookmarks)} bookmarks from {file_path}")
        return bookmarks
    except Exception as e:
        print(f"❌ Error loading bookmarks: {e}")
        return []


# --- ChromaDB + Embedding ---
def create_or_update_knowledge_base(bookmarks):
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    gemini_ef = GeminiEmbeddingFunction(api_key=GEMINI_API_KEY)

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=gemini_ef
    )

    print(f"✅ ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready. Current documents: {collection.count()}")

    existing_urls = set()
    try:
        existing_data = collection.get(include=['metadatas'])
        for meta in existing_data.get("metadatas", []):
            if meta.get("tweet_url"):
                existing_urls.add(meta["tweet_url"])
    except Exception as e:
        print(f"⚠️ Warning while checking existing documents: {e}")

    new_docs = []
    new_metas = []
    new_ids = []

    print("⏳ Preparing new tweets for embedding and ChromaDB storage...")
    for i, tweet in enumerate(tqdm(bookmarks, desc="Processing Tweets")):
        url = tweet.get("tweet_url")
        content = tweet.get("content", "")

        if not url or not content or content == "N/A" or url in existing_urls:
            continue

        new_docs.append(content)
        new_metas.append(tweet)
        new_ids.append(url)

    if new_docs:
        print(f"⏳ Adding {len(new_docs)} new tweets to ChromaDB...")
        collection.add(documents=new_docs, metadatas=new_metas, ids=new_ids)
        print(f"✅ Successfully added {len(new_docs)} new tweets to ChromaDB.")
    else:
        print("✅ No new tweets to add. ChromaDB is up to date.")

    print(f"📊 Total documents in collection '{CHROMA_COLLECTION_NAME}': {collection.count()}")
    return collection


# --- Main ---
if __name__ == "__main__":
    print("--- Starting Knowledge Base Builder ---")
    bookmarks = load_bookmarks(BOOKMARKS_FILE)
    if bookmarks:
        create_or_update_knowledge_base(bookmarks)
    print("--- Knowledge Base Builder Finished ---")
