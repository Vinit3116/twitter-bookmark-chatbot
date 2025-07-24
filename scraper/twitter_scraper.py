# scraper/twitter_scraper.py

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import time, json, os, re
from datetime import datetime

# Critical selectors to locate elements inside each tweet
SELECTORS = {
    "tweet_article": 'article[data-testid="tweet"]',
    "tweet_text_div": 'div[data-testid="tweetText"]',
    "tweet_url_link": 'a[href*="/status/"]',
    "author_name_span": 'div[data-testid="User-Name"] a[role="link"] div[dir="ltr"] span span',
    "author_handle_span": 'div[data-testid="User-Name"] a[href^="/"][tabindex="-1"] div[dir="ltr"] span',
    "tweet_date_time": 'a[href*="/status/"] time',
    "metrics_group_aria_label": 'div[role="group"][aria-label]',
}

def run_scraper():
    print("--- Starting X.com Bookmark Scraper ---")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://x.com/login")

        print("⚠️ Please log in manually...")
        time.sleep(45)

        print("✅ Login complete. Navigating to bookmarks...")
        page.goto("https://x.com/i/bookmarks", wait_until="domcontentloaded")
        page.wait_for_selector(SELECTORS["tweet_article"], timeout=15000)
        time.sleep(5)

        scraped_data = []
        last_tweet_count = 0
        scroll_attempts = 0
        max_scroll_attempts = 5
        extracted_tweet_urls = set()

        while scroll_attempts < max_scroll_attempts:
            page.mouse.wheel(0, 1500)
            time.sleep(3)
            tweets_on_page = page.query_selector_all(SELECTORS["tweet_article"])

            new_tweets_found = any(
                tweet.query_selector(SELECTORS["tweet_url_link"]).get_attribute("href") not in extracted_tweet_urls
                for tweet in tweets_on_page
                if tweet.query_selector(SELECTORS["tweet_url_link"])
            )

            if not new_tweets_found and len(tweets_on_page) == last_tweet_count:
                scroll_attempts += 1
                print(f"No new tweets loaded (attempt {scroll_attempts})")
            else:
                last_tweet_count = len(tweets_on_page)
                scroll_attempts = 0
                print(f"Loaded {len(tweets_on_page)} tweets...")

            for tweet_element in tweets_on_page:
                tweet_data = {}
                try:
                    tweet_url_el = tweet_element.query_selector(SELECTORS["tweet_url_link"])
                    tweet_data["tweet_url"] = (
                        f"https://x.com{tweet_url_el.get_attribute('href')}" if tweet_url_el else None
                    )

                    if not tweet_data["tweet_url"] or tweet_data["tweet_url"] in extracted_tweet_urls:
                        continue
                    extracted_tweet_urls.add(tweet_data["tweet_url"])

                    name_el = tweet_element.query_selector(SELECTORS["author_name_span"])
                    handle_el = tweet_element.query_selector(SELECTORS["author_handle_span"])
                    tweet_data["author_name"] = name_el.inner_text().strip() if name_el else "N/A"
                    tweet_data["author_handle"] = handle_el.inner_text().strip() if handle_el else "N/A"

                    time_el = tweet_element.query_selector(SELECTORS["tweet_date_time"])
                    if time_el:
                        dt = time_el.get_attribute("datetime")
                        tweet_data["tweet_date"] = (
                            datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime('%Y-%m-%d %H:%M:%S UTC')
                            if dt else time_el.inner_text().strip()
                        )
                    else:
                        tweet_data["tweet_date"] = "N/A"

                    text_el = tweet_element.query_selector(SELECTORS["tweet_text_div"])
                    raw_text = text_el.inner_text().strip() if text_el else "N/A"
                    tweet_data["content"] = re.sub(r'\s+', ' ', raw_text)

                    tweet_data.update({"replies": 0, "retweets": 0, "likes": 0, "views": 0})
                    metrics_el = tweet_element.query_selector(SELECTORS["metrics_group_aria_label"])
                    if metrics_el:
                        aria = metrics_el.get_attribute("aria-label")
                        if aria:
                            for key, pattern in {
                                "replies": r"(\d+)\s*repl(?:y|ies)",
                                "retweets": r"(\d+)\s*(?:reposts|retweets)",
                                "likes": r"(\d+)\s*likes?",
                                "views": r"(\d+)\s*views",
                            }.items():
                                match = re.search(pattern, aria, re.IGNORECASE)
                                if match:
                                    tweet_data[key] = int(match.group(1))

                    scraped_data.append(tweet_data)

                except Exception as e:
                    print(f"❌ Error on tweet: {e}")

        # Deduplicate and number tweets
        final = []
        seen_urls = set()
        for tweet in scraped_data:
            if tweet["tweet_url"] not in seen_urls:
                final.append(tweet)
                seen_urls.add(tweet["tweet_url"])
        for i, tweet in enumerate(final):
            tweet["tweet_number"] = i + 1

        # Save to JSON
        os.makedirs("scraper", exist_ok=True)
        with open("scraper/bookmarks.json", "w", encoding="utf-8") as f:
            json.dump(final, f, indent=4, ensure_ascii=False)
        print(f"✅ Scraped {len(final)} tweets to scraper/bookmarks.json")

        browser.close()

def load_bookmarks(filepath="scraper/bookmarks.json"):
    """Load saved tweet data from bookmarks.json."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bookmark file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    run_scraper()
