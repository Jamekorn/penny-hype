# ingestion/reddit_ingestor.py
import datetime as dt
from db.utils import get_engine
import pandas as pd
import praw
from dotenv import load_dotenv
import os

load_dotenv()

def get_reddit_client():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="penny-hype-ingestor"
    )

def fetch_reddit_posts(ticker, start, end, subreddit_list=None, limit=1000):
    reddit = get_reddit_client()
    if subreddit_list is None:
        subreddit_list = ["wallstreetbets", "stocks"]
    all_rows = []
    for sub in subreddit_list:
        subreddit = reddit.subreddit(sub)
        # e.g. use subreddit.search or subreddit.top with time filter
        # You’ll refine this, but v1: just grab posts containing ticker in title.
        for post in subreddit.search(ticker, limit=limit):
            created = dt.datetime.fromtimestamp(post.created_utc, dt.timezone.utc)
            if not (start <= created <= end):
                continue
            row = {
                "source_post_id": post.id,
                "ticker": ticker,
                "subreddit": sub,
                "author": str(post.author),
                "created_utc": created,
                "title": post.title,
                "body": post.selftext,
                "score": post.score,
                "num_comments": post.num_comments,
            }
            all_rows.append(row)
    return all_rows

def save_reddit_posts(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    engine = get_engine()
    df.to_sql("reddit_posts", engine, if_exists="append", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)
    rows = fetch_reddit_posts(args.ticker, start, end)
    save_reddit_posts(rows)
