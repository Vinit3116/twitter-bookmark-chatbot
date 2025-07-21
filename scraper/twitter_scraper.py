# scraper/twitter_scraper.py

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import time
import json
import os
import re # Used for parsing metrics from aria-label
from datetime import datetime # Used for formatting tweet date

# --- Playwright Selectors (CRITICAL: Verify these by inspecting X.com manually!) ---
# Twitter/X frequently updates its website, these selectors might change.
# Use your browser's inspect tool to find the correct data-testid or class names.
SELECTORS = {
    # General Selectors (for tweets)
    "tweet_article": 'article[data-testid="tweet"]', # Main container for each tweet

    # Within a tweet_article (most crucial for data extraction)
    "tweet_text_div": 'div[data-testid="tweetText"]', # Common selector for main tweet text content
    "tweet_url_link": 'a[href*="/status/"]', # Link that goes to the specific tweet status page (contains date info too)

    # --- REVISED AUTHOR SELECTORS (keep these as they are correct) ---
    "author_name_span": 'div[data-testid="User-Name"] a[role="link"] div[dir="ltr"] span span', # The span containing the author's display name
    "author_handle_span": 'div[data-testid="User-Name"] a[href^="/"][tabindex="-1"] div[dir="ltr"] span', # The span containing the author's @handle
    # ---------------------------------------------------------------------

    "tweet_date_time": 'a[href*="/status/"] time', # The <time> tag within the status link

    # --- REVERTED METRICS SELECTOR TO GROUP ARIA-LABEL (more robust for zero counts) ---
    "metrics_group_aria_label": 'div[role="group"][aria-label]', # Selects the div that contains the combined metrics in its aria-label
    # ---------------------------------------------------------------------------------
}

# The parse_metric function is no longer needed as we are directly parsing from aria-labels.
# It has been removed to simplify the code.

def run_scraper():
    print("--- Starting X.com Bookmark Scraper ---")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False) # Keep headless=False for visibility
        context = browser.new_context()

        page = context.new_page()
        page.goto("https://x.com/login")

        # Manual login instead of auto-filling username/password
        print("⚠️ Please log in manually within the browser window...")
        time.sleep(45)  # You can increase this if you need more time

        # Navigate to bookmarks
        print("✅ Login complete, navigating to bookmarks...")
        # Changed wait_until from "networkidle" to "domcontentloaded" for robustness
        page.goto("https://x.com/i/bookmarks", wait_until="domcontentloaded")
        # Explicitly wait for the first tweet article to appear, indicating content loaded
        page.wait_for_selector(SELECTORS["tweet_article"], timeout=15000)
        time.sleep(5) # Give a little extra time for bookmarks to load fully before scrolling

        # Scrape tweets
        print("🔍 Scraping bookmarks...")
        
        scraped_data = []
        last_tweet_count = 0
        scroll_attempts = 0
        max_scroll_attempts = 5 # Increased max scroll attempts to try and get more tweets

        # Use a set to keep track of extracted tweet URLs to avoid duplicates across scrolls
        extracted_tweet_urls = set() 

        while scroll_attempts < max_scroll_attempts:
            # Scroll down to load more content
            page.mouse.wheel(0, 1500) # Scroll down by 1500 pixels
            time.sleep(3) # Wait for new content to load

            # Get all tweet articles visible on the page
            tweets_on_page = page.query_selector_all(SELECTORS["tweet_article"])
            
            # Check if new tweets were loaded
            new_tweets_found_in_scroll = False
            for tweet_element in tweets_on_page:
                tweet_url_element = tweet_element.query_selector(SELECTORS["tweet_url_link"])
                current_tweet_url = tweet_url_element.get_attribute('href') if tweet_url_element else None
                if current_tweet_url and not current_tweet_url.startswith("http"):
                    current_tweet_url = f"https://x.com{current_tweet_url}"
                
                if current_tweet_url and current_tweet_url not in extracted_tweet_urls:
                    new_tweets_found_in_scroll = True
                    break # Found at least one new tweet

            if not new_tweets_found_in_scroll and len(tweets_on_page) == last_tweet_count:
                scroll_attempts += 1
                print(f"No new tweets loaded after scroll attempt {scroll_attempts}. Trying again...")
            else:
                last_tweet_count = len(tweets_on_page)
                scroll_attempts = 0 # Reset attempts if new content loads
                print(f"Loaded {len(tweets_on_page)} tweets after scrolling.")

            # Iterate through all tweets found so far and extract data
            for idx, tweet_element in enumerate(tweets_on_page):
                tweet_data = {}
                try:
                    # Extract Tweet URL and Date first, as the URL is often unique
                    tweet_url_element = tweet_element.query_selector(SELECTORS["tweet_url_link"])
                    tweet_data["tweet_url"] = tweet_url_element.get_attribute('href') if tweet_url_element else None

                    # Ensure URL is absolute
                    if tweet_data["tweet_url"] and not tweet_data["tweet_url"].startswith("http"):
                        tweet_data["tweet_url"] = f"https://x.com{tweet_data['tweet_url']}"

                    # Skip if this tweet was already scraped (useful for infinite scrolling)
                    if tweet_data["tweet_url"] in extracted_tweet_urls:
                        continue
                    if tweet_data["tweet_url"]: # Add to set only if it's a valid URL
                        extracted_tweet_urls.add(tweet_data["tweet_url"])

                    # Extract Author Name
                    author_name_element = tweet_element.query_selector(SELECTORS["author_name_span"])
                    tweet_data["author_name"] = author_name_element.inner_text().strip() if author_name_element else "N/A"

                    # Extract Author Handle
                    author_handle_element = tweet_element.query_selector(SELECTORS["author_handle_span"])
                    tweet_data["author_handle"] = author_handle_element.inner_text().strip() if author_handle_element else "N/A"

                    # Extract Tweet Date/Time
                    tweet_date_element = tweet_element.query_selector(SELECTORS["tweet_date_time"])
                    if tweet_date_element:
                        datetime_attr = tweet_date_element.get_attribute('datetime')
                        if datetime_attr:
                            # Parse ISO 8601 format to a more readable string
                            tweet_data["tweet_date"] = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S UTC')
                        else:
                            tweet_data["tweet_date"] = tweet_date_element.inner_text().strip() # Fallback to inner text if datetime attr is missing
                    else:
                        tweet_data["tweet_date"] = "N/A"

                    # Extract Tweet Text (main content)
                    tweet_text_div = tweet_element.query_selector(SELECTORS["tweet_text_div"])
                    # Use regex to clean up potential extra newlines/spaces if needed
                    content_raw = tweet_text_div.inner_text().strip() if tweet_text_div else "N/A"
                    tweet_data["content"] = re.sub(r'\s+', ' ', content_raw).strip() # Replace multiple spaces/newlines with single space


                    # --- REVISED METRICS EXTRACTION USING GROUP ARIA-LABEL AND ROBUST REGEX ---
                    tweet_data["replies"] = 0
                    tweet_data["retweets"] = 0
                    tweet_data["likes"] = 0
                    tweet_data["views"] = 0

                    metrics_group = tweet_element.query_selector(SELECTORS["metrics_group_aria_label"])
                    if metrics_group:
                        aria_label_text = metrics_group.get_attribute('aria-label')
                        if aria_label_text:
                            # Regex patterns to find numbers followed by specific keywords, making them optional
                            # If a pattern doesn't match, the metric remains 0 (its initial value)
                            
                            # Replies: Look for "X repl(y|ies)"
                            replies_match = re.search(r'(\d+)\s*repl(?:y|ies)', aria_label_text, re.IGNORECASE)
                            if replies_match:
                                tweet_data["replies"] = int(replies_match.group(1))

                            # Retweets/Reposts: Look for "X reposts" or "X retweets"
                            reposts_match = re.search(r'(\d+)\s*(?:reposts|retweets)', aria_label_text, re.IGNORECASE)
                            if reposts_match:
                                tweet_data["retweets"] = int(reposts_match.group(1))

                            # Likes: Look for "X like(s)"
                            likes_match = re.search(r'(\d+)\s*likes?', aria_label_text, re.IGNORECASE) # 's?' for optional 's'
                            if likes_match:
                                tweet_data["likes"] = int(likes_match.group(1))

                            # Views: Look for "X views"
                            views_match = re.search(r'(\d+)\s*views', aria_label_text, re.IGNORECASE)
                            if views_match:
                                tweet_data["views"] = int(views_match.group(1))
                    # -----------------------------------------------------------------------


                    scraped_data.append(tweet_data)

                    print(f"\n--- Tweet {len(scraped_data)} ---")
                    print(f"URL: {tweet_data['tweet_url']}")
                    print(f"Author: {tweet_data['author_name']} ({tweet_data['author_handle']})")
                    print(f"Date: {tweet_data['tweet_date']}")
                    print(f"Content: {tweet_data['content'][:150]}...") # Print first 150 chars
                    print(f"Metrics: R:{tweet_data['replies']} RT:{tweet_data['retweets']} L:{tweet_data['likes']} V:{tweet_data['views']}")
                    print("-" * 50)

                except PlaywrightTimeoutError as pte:
                    print(f"❌ Playwright Timeout Error on tweet {idx + 1}: {pte}")
                    # This might happen if a sub-selector doesn't appear quickly
                except Exception as e:
                    print(f"❌ Error extracting tweet {idx + 1} data: {e}")
                    print(f"  Attempted Tweet HTML: {tweet_element.inner_html()[:200]}...") # Print a snippet of the problematic HTML

        # Add a tweet_number to the final de-duplicated list
        # This loop is moved outside the while loop to ensure numbers are sequential after all scrolling and de-duplication
        final_scraped_tweets = []
        seen_urls_final = set()
        for tweet in scraped_data:
            if tweet.get('tweet_url') and tweet['tweet_url'] not in seen_urls_final:
                final_scraped_tweets.append(tweet)
                seen_urls_final.add(tweet['tweet_url'])
        
        # Assign tweet_number after final de-duplication
        for i, tweet in enumerate(final_scraped_tweets):
            tweet['tweet_number'] = i + 1


        # Save to JSON
        output_file = "scraper/bookmarks.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True) # Ensure directory exists

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_scraped_tweets, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Successfully scraped {len(final_scraped_tweets)} unique tweets and saved to {output_file}")

        # Close browser
        browser.close()

if __name__ == "__main__":
    run_scraper()
