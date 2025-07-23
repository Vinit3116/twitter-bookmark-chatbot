# embeddings/embedder.py

import os
import json
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in .env file. Please set it to proceed.")
    exit("Exiting: GEMINI_API_KEY is required.")

# Configure the Google Generative AI client for embeddings
genai.configure(api_key=GEMINI_API_KEY)

# Path to your scraped bookmarks JSON file
BOOKMARKS_FILE = 'scraper/bookmarks.json'
# Name for your ChromaDB collection
CHROMA_COLLECTION_NAME = 'twitter_bookmarks'
# Directory to store ChromaDB data (relative to the project root)
CHROMA_DB_PATH = './chroma_db' 

# Gemini Embedding Model
EMBEDDING_MODEL_GEMINI = 'models/embedding-001' # This outputs 768-dimensional embeddings
EMBEDDING_DIMENSION = 768 # Explicitly set for this model

# --- Custom Gemini Embedding Function for ChromaDB ---
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = EMBEDDING_MODEL_GEMINI):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        logging.info(f"Gemini embedding model '{self.model_name}' configured for documents.")

    def __call__(self, texts: list[str]):
        """Generates embeddings for a list of texts using the Gemini API."""
        embeddings = []
        for text in texts:
            try:
                response = genai.embed_content(
                    model=self.model_name, 
                    content=[text], 
                    task_type="retrieval_document"
                )
                embeddings.append(response['embedding'][0])
            except Exception as e:
                logging.error(f"Error generating embedding for text: '{text[:50]}...': {e}")
                embeddings.append([0.0] * EMBEDDING_DIMENSION) # Fallback with correct dimension
        return embeddings

# Initialize the custom embedding function
gemini_ef = GeminiEmbeddingFunction(api_key=GEMINI_API_KEY)

# --- Core Functions ---

def load_bookmarks(file_path):
    """Loads bookmarked tweets from a JSON file."""
    if not os.path.exists(file_path):
        logging.error(f"Bookmarks file not found: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            bookmarks = json.load(f)
        logging.info(f"Loaded {len(bookmarks)} bookmarks from {file_path}")
        return bookmarks
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading bookmarks: {e}")
        return []

def create_or_update_knowledge_base(bookmarks):
    """
    Creates or loads a ChromaDB collection and adds new tweet data and embeddings.
    It intelligently avoids re-adding tweets that are already present in the collection
    by checking their 'tweet_url'.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=gemini_ef # Pass our custom Gemini embedding function here
    )
    logging.info(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready. Current documents: {collection.count()}")

    existing_tweet_urls = set()
    if collection.count() > 0:
        try:
            existing_docs = collection.get(
                where={"tweet_url": {"$ne": None}}, 
                include=['metadatas']
            )
            for doc_metadata in existing_docs.get('metadatas', []):
                if 'tweet_url' in doc_metadata:
                    existing_tweet_urls.add(doc_metadata['tweet_url'])
            logging.info(f"Found {len(existing_tweet_urls)} existing tweet URLs in ChromaDB.")
        except Exception as e:
            logging.warning(f"Could not retrieve existing documents from ChromaDB: {e}. Proceeding as if no existing tweets.")
            existing_tweet_urls = set()

    documents_to_add = [] 
    metadatas_to_add = []
    ids_to_add = []       
    
    logging.info("Preparing new tweets for embedding and ChromaDB storage...")
    for tweet in tqdm(bookmarks, desc="Processing Tweets"):
        tweet_url = tweet.get('tweet_url')
        content_text = tweet.get('content', '')

        if not tweet_url or not content_text or content_text == "N/A" or tweet_url in existing_tweet_urls:
            continue
        
        documents_to_add.append(content_text)
        metadatas_to_add.append(tweet)
        ids_to_add.append(tweet_url)

    if documents_to_add:
        try:
            logging.info(f"Adding {len(documents_to_add)} new tweets to ChromaDB...")
            collection.add(
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            logging.info(f"Successfully added {len(documents_to_add)} new tweets to ChromaDB.")
        except Exception as e:
            logging.error(f"ERROR: Failed to add documents to ChromaDB: {e}")
    else:
        logging.info("No new tweets to add to ChromaDB.")

    logging.info(f"Total documents in ChromaDB collection '{CHROMA_COLLECTION_NAME}': {collection.count()}")
    return collection

# --- Main Execution Block ---
if __name__ == "__main__":
    logging.info("Starting knowledge base creation process...")
    
    all_bookmarks = load_bookmarks(BOOKMARKS_FILE)

    if all_bookmarks:
        chroma_collection = create_or_update_knowledge_base(all_bookmarks)
    
    logging.info("Knowledge base creation process finished.")

