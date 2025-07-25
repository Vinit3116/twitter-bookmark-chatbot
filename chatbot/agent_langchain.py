import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# --------------------------
# 🔍 DOMAIN-SPECIFIC SYNONYMS
# --------------------------
# Mapping of topics to relevant keywords for high-precision filtering.
TOPIC_SYNONYMS = {
    "cricket": ["cricket", "ipl", "yorker", "wicket", "innings", "overs", "bowled", "siraj", "stokes", "jayasuriya"],
    "siraj": ["siraj", "mohammad siraj"],
    "neeraj chopra": ["neeraj", "chopra", "javelin", "athletics"],
    "javelin": ["javelin", "neeraj", "chopra", "athletics"],
    "athletics": ["athletics", "track", "field", "sprint", "relay", "javelin"],
    "hockey": ["hockey"],
    "football": ["football", "soccer"],
    "politics": [
        "politics", "politician", "vice president", "president", "parliament", "government", "minister",
        "dhankhar", "jagdeep", "rajya sabha", "lok sabha", "election", "office", "sealed"
    ],
    "investment": [
        "investment", "investing", "investor", "stock", "stocks", "share", "mutual fund", "funds", "finance", "financial", "advisor", "equity", "returns"
    ]
}

# --------------------------
# ✨ SENTIMENT & AI KEYWORDS
# --------------------------
POSITIVE_WORDS = [
    "great", "improve", "best", "success", "win", "awesome", "good", "happy", "love", "excellent", "positive",
    "cool", "brilliant", "enjoy", "relief", "historic", "inspire", "achieve", "vision", "launch", "plan", "goal", "future", "ambitious"
]
AI_KEYWORDS = [
    "ai", "artificial intelligence", "xai", "openai", "gemini", "ai agent", "ai agents"
]

# --------------------------
# 🔍 UTILITY MATCHERS
# --------------------------

# Checks if tweet mentions AI
def is_ai_related(text):
    return bool(text and any(re.search(rf"\b{k}\b", text.lower()) for k in AI_KEYWORDS))

# Checks if tweet has a positive tone
def has_positive(text):
    return bool(text and any(re.search(rf'\b{w}\b', text.lower()) for w in POSITIVE_WORDS))

# Expands topic to all synonyms
def expand_synonyms(topic):
    kw = topic.lower().strip()
    out = set([kw])
    for k, syns in TOPIC_SYNONYMS.items():
        if kw == k or kw in syns:
            out.update(syns)
    return list(out)

# --------------------------
# 🧠 FILTER DETECTION FROM QUESTION
# --------------------------
def detect_and_extract_filters(question: str):
    filters = {}
    lower_q = question.lower()

    # Likes and views thresholds
    likes_match = re.search(r'(\d{1,8})\s*\+?\s*likes?', lower_q)
    if likes_match:
        filters['likes'] = int(likes_match.group(1))

    views_match = re.search(r'(\d{1,10})\s*\+?\s*views?', lower_q)
    if views_match:
        filters['views'] = int(views_match.group(1))

    # Topic keyword
    topic_match = re.search(r'(about|related to|mentioning)\s+([\w\s#@.\'-]+)', lower_q)
    if topic_match:
        filters['topic'] = topic_match.group(2).strip()

    # Sentiment
    if "positive" in lower_q:
        filters["sentiment"] = "positive"

    # Recency
    if "most recent" in lower_q or "latest" in lower_q:
        filters["recency"] = True

    # Most liked
    if "most liked" in lower_q or "top liked" in lower_q:
        filters["most_liked"] = True

    # Summary/overview queries
    if any(word in lower_q for word in ["summarize", "main topic", "what topics", "themes"]):
        filters["summarize"] = True

    # Ranking/filtering by users
    if any(word in lower_q for word in ["most-bookmarked", "most bookmarked", "top users", "most frequent users"]):
        filters["ranking"] = "user"

    return filters

# --------------------------
# 🧹 FILTER DOCUMENTS BASED ON CONDITIONS
# --------------------------
def filter_documents(documents, filters, question=""):
    filtered = documents

    # Filter by likes/views if not asking for most-liked tweet
    if "likes" in filters and not filters.get("most_liked"):
        filtered = [d for d in filtered if int(d.metadata.get("likes", 0)) >= filters["likes"]]
    if "views" in filters:
        filtered = [d for d in filtered if int(d.metadata.get("views", 0)) >= filters["views"]]

    have_positive = filters.get("sentiment") == "positive"
    wants_ai = any(re.search(rf'\b{k}\b', question.lower()) for k in AI_KEYWORDS)

    # Special case: user asked for positive AI tweets
    if have_positive and wants_ai:
        strict = [d for d in filtered if has_positive(d.page_content) and is_ai_related(d.page_content)]
        if strict:
            return ("STRICT_AI", strict)
        fallback = [d for d in filtered if is_ai_related(d.page_content)]
        if fallback:
            return ("FALLBACK_AI", fallback)
        return ("NO_AI", [])

    # General topic filtering
    if "topic" in filters:
        topic = filters["topic"].lower()
        synonyms = expand_synonyms(topic)
        results = []
        for d in filtered:
            all_chunks = [
                d.page_content.lower(),
                str(d.metadata.get("content", "")).lower(),
                str(d.metadata.get("author", "")).lower(),
                str(d.metadata.get("author_handle", "")).lower()
            ]
            for syn in synonyms:
                pat = re.compile(rf'\b{re.escape(syn)}\b', re.IGNORECASE)
                if any(pat.search(chunk) for chunk in all_chunks):
                    results.append(d)
                    break
        return results

    # Sentiment-only filtering
    if have_positive:
        filtered = [d for d in filtered if has_positive(d.page_content)]

    return filtered

# --------------------------
# ⭐ GET TWEET WITH MAX LIKES FOR TOPIC
# --------------------------
def get_most_liked_tweet(documents, topic_search=None):
    candidates = documents
    if topic_search:
        synonyms = expand_synonyms(topic_search)
        filtered = []
        is_ai_query = topic_search in AI_KEYWORDS
        for d in documents:
            all_chunks = [
                d.page_content.lower(),
                str(d.metadata.get("content", "")).lower(),
                str(d.metadata.get("author", "")).lower(),
                str(d.metadata.get("author_handle", "")).lower()
            ]
            match = False
            for syn in synonyms:
                pat = re.compile(rf'\b{re.escape(syn)}\b', re.IGNORECASE)
                if any(pat.search(chunk) for chunk in all_chunks):
                    match = True
                    break
            if is_ai_query and not match and is_ai_related(d.page_content):
                match = True
            if match:
                filtered.append(d)
        candidates = filtered
    return max(candidates, key=lambda d: int(d.metadata.get("likes", 0)), default=None) if candidates else None

# --------------------------
# 🧑 TOP USERS BY BOOKMARK FREQUENCY
# --------------------------
def get_most_bookmarked_users(documents, top_n=5):
    from collections import Counter
    counts = Counter(d.metadata.get("author", "Unknown") for d in documents)
    return counts.most_common(top_n)

# --------------------------
# ⏱️ MOST RECENT BOOKMARKED TWEET (assumes sorted input)
# --------------------------
def get_most_recent_tweet(documents):
    return documents[0] if documents else None

# --------------------------
# 🤖 MAIN AGENT LOGIC
# --------------------------
class SmartAgent:
    def __init__(self, retriever, llm, all_documents):
        self.retriever = retriever
        self.llm = llm
        self.all_docs = all_documents

    def invoke(self, inputs):
        question = inputs["question"]
        filters = detect_and_extract_filters(question)

        # Most liked tweet logic
        if filters.get("most_liked"):
            topic_search = None
            for k in list(AI_KEYWORDS) + list(TOPIC_SYNONYMS.keys()):
                if re.search(rf'\b{k}\b', question.lower()):
                    topic_search = k
                    break
            tweet = get_most_liked_tweet(self.all_docs, topic_search)
            if tweet and int(tweet.metadata.get("likes", 0)) > 0:
                return (
                    f'The most liked tweet{" about " + topic_search if topic_search else ""} is:\n'
                    f'"{tweet.page_content}" — {tweet.metadata.get("author", "")}, '
                    f'{tweet.metadata.get("likes", 0)} likes, {tweet.metadata.get("views", 0)} views\n'
                    f'Date: {tweet.metadata.get("date", "")}\nURL: {tweet.metadata.get("tweet_url", "")}'
                )
            return "No matching tweet found."

        # Positive + AI filtering logic
        if filters.get("sentiment") == "positive" and any(re.search(rf'\b{k}\b', question.lower()) for k in AI_KEYWORDS):
            kind, docs = filter_documents(self.all_docs, filters, question)
            if kind == "STRICT_AI":
                return "\n".join(
                    f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
                    f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
                    f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                    for d in docs[:3]
                )
            elif kind == "FALLBACK_AI":
                msg = "No positive tweets about AI found, but here are tweets mentioning AI:\n"
                return msg + "\n".join(
                    f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
                    f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
                    f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                    for d in docs[:3]
                )
            return "No tweets about AI found."

        # Topic/entity filtering
        if "topic" in filters:
            docs = filter_documents(self.all_docs, filters, question)
            if not docs:
                return f"No bookmarks found related to '{filters['topic']}'."
            return "\n".join(
                f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
                f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
                f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                for d in docs[:5]
            )

        # Top bookmarked users
        if filters.get("ranking") == "user":
            users = get_most_bookmarked_users(self.all_docs)
            return "You most frequently bookmark these users:\n" + "\n".join(
                f"{i+1}. {name} ({count} times)" for i, (name, count) in enumerate(users)
            )

        # Most recent tweet
        if filters.get("recency"):
            doc = get_most_recent_tweet(self.all_docs)
            if doc:
                return (
                    f"The most recent bookmarked tweet is:\n"
                    f"\"{doc.page_content}\" — {doc.metadata.get('author', '')}\n"
                    f"Date: {doc.metadata.get('date', '')}\nURL: {doc.metadata.get('tweet_url', '')}"
                )
            return "No tweet date information found in your bookmarks."

        # Likes/views/sentiment filtering
        if any(k in filters for k in ["likes", "views", "sentiment"]):
            docs = filter_documents(self.all_docs, filters, question)
            if not docs:
                return "No relevant bookmarks found."
            return "\n".join(
                f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
                f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
                f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                for d in sorted(docs, key=lambda x: int(x.metadata.get("likes", 0)), reverse=True)[:5]
            )

        # Summarization mode
        if filters.get("summarize"):
            docs = self.retriever.get_relevant_documents(question)
            context = "\n".join([d.page_content for d in docs][:20])
            summ_prompt = (
                "From the following tweets, extract and summarize the main topics, hashtags, and common themes. "
                "Present a concise, bulleted list (optionally ranked by frequency/popularity).\n"
                f"Tweets:\n{context}"
            )
            return self.llm.invoke(summ_prompt).content

        # Fallback to retrieval if no filter matched
        docs = self.retriever.get_relevant_documents(question)
        if not docs:
            return "No relevant bookmarks found."
        return "\n".join(
            f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
            f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
            f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
            for d in docs[:5]
        )

# --------------------------
# 🏗️ AGENT BUILDER FUNCTION
# --------------------------
def build_agent(collection_name, embedding_function, persist_directory, all_documents):
    retriever = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    ).as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return SmartAgent(retriever, llm, all_documents)
