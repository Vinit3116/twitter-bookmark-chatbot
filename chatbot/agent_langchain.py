import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma

# 1. Env load
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your .env file!")

# 2. Model/DB settings
EMBED_MODEL = "models/embedding-001"
LLM_MODEL = "models/gemini-2.5-flash"
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "twitter_bookmarks"

# 3. Embedding & vectorstore setup
embedding_function = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=GEMINI_API_KEY
)
vectordb = Chroma(
    collection_name=CHROMA_COLLECTION,
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 4. LLM + memory (NOTE output_key="answer" is added below!)
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.5,
    google_api_key=GEMINI_API_KEY
)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"      # <-- This is required for modern LangChain (memory fix!)
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

if __name__ == "__main__":
    print("Gemini+LangChain Memory Chatbot")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        if not user_input:
            print("Chatbot: Please enter a query.")
            continue
        try:
            result = qa_chain.invoke({"question": user_input})
            print("\nChatbot:", result["answer"])
            if result.get("source_documents"):
                print("--- Relevant bookmarks:")
                for doc in result["source_documents"]:
                    print("-", doc.page_content[:120].replace('\n',' '))
            print()
        except Exception as e:
            print(f"Error: {e}\n")
