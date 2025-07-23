import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
assert GEMINI_API_KEY, "Set GEMINI_API_KEY in .env!"

genai.configure(api_key=GEMINI_API_KEY)

## ChromaDB
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "twitter_bookmarks"
EMBED_MODEL = "models/embedding-001"
GEN_MODEL = "models/gemini-2.5-flash"    # or models/gemini-1.5-flash if that's your tier

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

def get_query_embedding(query):
    return genai.embed_content(
        model=EMBED_MODEL,
        content=[query],
        task_type="retrieval_document"
    )["embedding"][0]

def get_relevant_tweets(query, n=3):
    embedding = get_query_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["documents", "metadatas"]
    )
    # Return a list of tweet dicts for display
    return [
        {"text": doc, "meta": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

chat_model = genai.GenerativeModel(GEN_MODEL)

def qa_prompt(user_query, context_tweets):
    context = "\n\n".join([
        f"URL: {t['meta'].get('tweet_url', 'N/A')}\nAuthor: {t['meta'].get('author_name', 'N/A')} (@{t['meta'].get('author_handle', 'N/A')})\nContent: {t['text']}"
        for t in context_tweets
    ])
    return f"""You are a helpful assistant summarizing the user's Twitter bookmarks.
Do not make up information. Use only the following tweets as sources.

---
{context}
---

User question: {user_query}
Answer:"""

if __name__ == "__main__":
    print("Twitter Bookmark Gemini Chatbot")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            print("Please ask a question about your bookmarks.")
            continue
        try:
            context = get_relevant_tweets(user_input, n=3)
            prompt = qa_prompt(user_input, context)
            response = chat_model.generate_content(prompt)
            try:
                out = response.candidates[0].content.parts[0].text
            except Exception:
                out = "Sorry, I didn't get a valid answer this time."
            print("\nChatbot:", out)
            if context:
                print("--- Relevant bookmarks:")
                for t in context:
                    print("-", t["text"][:120])
            print()
        except Exception as e:
            print("Error:", e)
            print()
