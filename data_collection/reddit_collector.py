import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from loguru import logger
from config.settings import settings

class RedditCollector:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=settings.REDDIT_CONFIG["client_id"],
            client_secret=settings.REDDIT_CONFIG["client_secret"],
            user_agent=settings.REDDIT_CONFIG["user_agent"]
        )
        
    def collect_posts(self, ticker: str, subreddits: List[str] = None, 
                     limit_per_sub: int = 100) -> pd.DataFrame:
        """Collect posts mentioning ticker from specified subreddits"""
        if subreddits is None:
            subreddits = ["wallstreetbets", "stocks", "investing", "pennystocks"]
        
        posts = []
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                for post in subreddit.search(f"${ticker} OR {ticker}", limit=limit_per_sub):
                    posts.append({
                        "id": post.id,
                        "title": post.title,
                        "content": post.selftext,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc),
                        "author": str(post.author),
                        "subreddit": subreddit_name,
                        "url": post.url,
                        "ticker": ticker
                    })
            except Exception as e:
                logger.error(f"Error collecting from {subreddit_name}: {e}")
        
        return pd.DataFrame(posts)
    
    def collect_comments(self, ticker: str, limit: int = 500) -> pd.DataFrame:
        """Collect comments mentioning ticker"""
        comments = []
        try:
            for comment in self.reddit.subreddit("all").comments(limit=limit):
                if ticker.lower() in comment.body.lower() or f"${ticker}" in comment.body:
                    comments.append({
                        "id": comment.id,
                        "content": comment.body,
                        "score": comment.score,
                        "created_utc": datetime.fromtimestamp(comment.created_utc),
                        "author": str(comment.author),
                        "parent_id": comment.parent_id,
                        "ticker": ticker
                    })
        except Exception as e:
            logger.error(f"Error collecting comments: {e}")
        
        return pd.DataFrame(comments)

# Test the collector
if __name__ == "__main__":
    collector = RedditCollector()
    posts_df = collector.collect_posts("GME", limit_per_sub=10)
    print(f"Collected {len(posts_df)} posts")
    print(posts_df.head())