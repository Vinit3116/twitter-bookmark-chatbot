import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# --------------------------
# 🔍 DOMAIN-SPECIFIC SYNONYMS
# --------------------------
TOPIC_SYNONYMS = {
    "cricket": ["cricket", "ipl", "yorker", "wicket", "innings", "overs", "bowled", "siraj", "stokes", "jayasuriya", "pant", "rishabh pant"],
    "siraj": ["siraj", "mohammad siraj", "mohammed siraj"],
    "rishabh pant": ["pant", "rishabh pant"],
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

POSITIVE_WORDS = [
    "great", "improve", "best", "success", "win", "awesome", "good", "happy", "love", "excellent", "positive",
    "cool", "brilliant", "enjoy", "relief", "historic", "inspire", "achieve", "vision", "launch", "plan", "goal", "future", "ambitious"
]
AI_KEYWORDS = [
    "ai", "artificial intelligence", "xai", "openai", "gemini", "ai agent", "ai agents"
]

# ------- Matching & Helpers --------
def is_ai_related(text):
    return bool(text and any(re.search(rf"\b{k}\b", text.lower()) for k in AI_KEYWORDS))

def has_positive(text):
    return bool(text and any(re.search(rf'\b{w}\b', text.lower()) for w in POSITIVE_WORDS))

def expand_synonyms(topic):
    kw = topic.lower().strip()
    out = set([kw])
    for k, syns in TOPIC_SYNONYMS.items():
        if kw == k or kw in syns:
            out.update(syns)
    return list(out)

def strict_entity_filter(tweets, entity_synonyms):
    """Return only tweets where synonyms appear as whole words in main text/content fields."""
    synonym_patterns = [re.compile(rf'\b{re.escape(syn)}\b', re.IGNORECASE) for syn in entity_synonyms]
    results = []
    for t in tweets:
        # Combine both page_content and content fields, force str+lower
        text_fields = ((t.page_content or "") + " " + str(t.metadata.get("content", "") or "")).lower()
        if any(pat.search(text_fields) for pat in synonym_patterns):
            results.append(t)
    return results

def detect_and_extract_filters(question: str):
    filters = {}
    lower_q = question.lower()
    likes_match = re.search(r'(\d{1,8})\s*\+?\s*likes?', lower_q)
    if likes_match:   filters['likes'] = int(likes_match.group(1))
    views_match = re.search(r'(\d{1,10})\s*\+?\s*views?', lower_q)
    if views_match:   filters['views'] = int(views_match.group(1))
    topic_match = re.search(r'(about|related to|mentioning)\s+([\w\s#@.\'-]+)', lower_q)
    if topic_match:   filters['topic'] = topic_match.group(2).strip()
    if "positive" in lower_q: filters["sentiment"] = "positive"
    if "most recent" in lower_q or "latest" in lower_q: filters["recency"] = True
    if "most liked" in lower_q or "top liked" in lower_q: filters["most_liked"] = True
    if any(word in lower_q for word in ["summarize", "main topic", "what topics", "themes"]): filters["summarize"] = True
    if any(word in lower_q for word in ["most-bookmarked", "most bookmarked", "top users", "most frequent users"]): filters["ranking"] = "user"
    return filters

def filter_documents(documents, filters, question=""):
    filtered = documents
    if "likes" in filters and not filters.get("most_liked"):
        filtered = [d for d in filtered if int(d.metadata.get("likes", 0)) >= filters["likes"]]
    if "views" in filters:
        filtered = [d for d in filtered if int(d.metadata.get("views", 0)) >= filters["views"]]

    have_positive = filters.get("sentiment") == "positive"
    wants_ai = any(re.search(rf'\b{k}\b', question.lower()) for k in AI_KEYWORDS)

    # Special: positive AI tweets
    if have_positive and wants_ai:
        strict = [d for d in filtered if has_positive(d.page_content) and is_ai_related(d.page_content)]
        if strict: return ("STRICT_AI", strict)
        fallback = [d for d in filtered if is_ai_related(d.page_content)]
        if fallback: return ("FALLBACK_AI", fallback)
        return ("NO_AI", [])

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

    if have_positive:
        filtered = [d for d in filtered if has_positive(d.page_content)]
    return filtered

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
            if match: filtered.append(d)
        candidates = filtered
    return max(candidates, key=lambda d: int(d.metadata.get("likes", 0)), default=None) if candidates else None

def get_most_bookmarked_users(documents, top_n=5):
    from collections import Counter
    counts = Counter(d.metadata.get("author", "Unknown") for d in documents)
    return counts.most_common(top_n)

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
        search_space = inputs.get("search_space")
        docs = search_space if (search_space is not None and len(search_space) > 0) else self.all_docs
        filters = detect_and_extract_filters(question)

        # STRICT ENTITY/TOPIC MENTION HANDLER
        entity_asked = None
        # Named topic/entity (Siraj, Pant, etc)
        for k in list(TOPIC_SYNONYMS.keys()):
            if re.search(rf'\b{k}\b', question.lower()):
                entity_asked = k
                break
        # "mention of X", "about X", etc
        mention_reg = re.search(r"(?:mention(?:ed|ing)?|about)\s+([\w\s'-]+)", question.lower())
        if mention_reg:
            entity_asked = mention_reg.group(1).strip()
        if entity_asked:
            entity_synonyms = expand_synonyms(entity_asked)
            strict_matches = strict_entity_filter(docs, entity_synonyms)
            if not strict_matches:
                return f"No bookmarks found mentioning {entity_asked}.", []
            return (
                "\n".join(
                    f'- "{d.page_content}" — {d.metadata.get("author", "")}, {d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                    for d in strict_matches[:5]
                ),
                strict_matches[:5]
            )

        # Most liked tweet
        if filters.get("most_liked"):
            topic_search = None
        for k in list(AI_KEYWORDS) + list(TOPIC_SYNONYMS.keys()):
            if re.search(rf'\b{k}\b', question.lower()):
                topic_search = k
                break
        tweet = get_most_liked_tweet(docs, topic_search)
        if tweet and int(tweet.metadata.get("likes", 0)) > 0:
            return (
            f'The most liked tweet{" about " + topic_search if topic_search else ""} is:\n'
            f'"{tweet.page_content}" — {tweet.metadata.get("author", "")}, '
            f'{tweet.metadata.get("likes", 0)} likes, {tweet.metadata.get("views", 0)} views\n'
            f'Date: {tweet.metadata.get("date", "")}\nURL: {tweet.metadata.get("tweet_url", "")}',
            [tweet]
        )
        return "No matching tweet found.", []


        # Positive + AI filtering logic
        if filters.get("sentiment") == "positive" and any(re.search(rf'\b{k}\b', question.lower()) for k in AI_KEYWORDS):
            kind, result_docs = filter_documents(docs, filters, question)
            if kind == "STRICT_AI":
                return (
                    "\n".join(
                        f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
                        f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
                        f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                        for d in result_docs[:3]
                    ),
                    result_docs[:3]
                )
            elif kind == "FALLBACK_AI":
                msg = "No positive tweets about AI found, but here are tweets mentioning AI:\n"
                return (
                    msg + "\n".join(
                        f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
                        f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
                        f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                        for d in result_docs[:3]
                    ),
                    result_docs[:3]
                )
            return "No tweets about AI found.", []

        # Topic/entity filtering (for non-named-entity broad topics; e.g. "cricket", "politics", "investment")
        if "topic" in filters:
            result_docs = filter_documents(docs, filters, question)
            if not result_docs:
                return f"No bookmarks found related to '{filters['topic']}'.", []
            return (
                "\n".join(
                    f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
                    f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
                    f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                    for d in result_docs[:5]
                ),
                result_docs[:5]
            )

        # Top bookmarked users
        if filters.get("ranking") == "user":
            users = get_most_bookmarked_users(docs)
            return (
                "You most frequently bookmark these users:\n" + "\n".join(
                    f"{i+1}. {name} ({count} times)" for i, (name, count) in enumerate(users)
                ),
                []
            )

        # Most recent tweet
        if filters.get("recency"):
            doc = get_most_recent_tweet(docs)
            if doc:
                return (
                    f"The most recent bookmarked tweet is:\n"
                    f"\"{doc.page_content}\" — {doc.metadata.get('author', '')}\n"
                    f"Date: {doc.metadata.get('date', '')}\nURL: {doc.metadata.get('tweet_url', '')}",
                    [doc]
                )
            return "No tweet date information found in your bookmarks.", []

        # Likes/views/sentiment filtering
        if any(k in filters for k in ["likes", "views", "sentiment"]):
            result_docs = filter_documents(docs, filters, question)
            if not result_docs:
                return "No relevant bookmarks found.", []
            return (
                "\n".join(
                    f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
                    f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
                    f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                    for d in sorted(result_docs, key=lambda x: int(x.metadata.get("likes", 0)), reverse=True)[:5]
                ),
                sorted(result_docs, key=lambda x: int(x.metadata.get("likes", 0)), reverse=True)[:5]
            )

        # Summarization
        if filters.get("summarize"):
            result_docs = self.retriever.get_relevant_documents(question)
            context = "\n".join([d.page_content for d in result_docs][:20])
            summ_prompt = (
                "From the following tweets, extract and summarize the main topics, hashtags, and common themes. "
                "Present a concise, bulleted list (optionally ranked by frequency/popularity).\n"
                f"Tweets:\n{context}"
            )
            summary = self.llm.invoke(summ_prompt).content
            return summary, result_docs[:5]

        # Fallback
        result_docs = self.retriever.get_relevant_documents(question)
        if not result_docs:
            return "No relevant bookmarks found.", []
        return (
            "\n".join(
                f'- "{d.page_content}" — {d.metadata.get("author", "")}, '
                f'{d.metadata.get("likes", 0)} likes, {d.metadata.get("views", 0)} views\n  '
                f'Date: {d.metadata.get("date", "")}\n  URL: {d.metadata.get("tweet_url", "")}'
                for d in result_docs[:5]
            ),
            result_docs[:5]
        )

def build_agent(collection_name, embedding_function, persist_directory, all_documents):
    retriever = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    ).as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return SmartAgent(retriever, llm, all_documents)
