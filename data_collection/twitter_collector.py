import tweepy
import pandas as pd
from datetime import datetime
from typing import List, Dict
from loguru import logger
from config.settings import settings

class TwitterCollector:
    def __init__(self):
        self.client = tweepy.Client(
            bearer_token=settings.TWITTER_CONFIG["bearer_token"],
            consumer_key=settings.TWITTER_CONFIG["api_key"],
            consumer_secret=settings.TWITTER_CONFIG["api_secret"]
        )
    
    def collect_tweets(self, ticker: str, max_results: int = 100) -> pd.DataFrame:
        """Collect tweets mentioning ticker"""
        tweets = []
        try:
            # Search for tweets with $TICKER or #TICKER
            query = f"${ticker} OR #{ticker} -is:retweet"
            
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=["created_at", "public_metrics", "author_id"],
                expansions=["author_id"]
            )
            
            if response.data:
                for tweet in response.data:
                    tweets.append({
                        "id": tweet.id,
                        "content": tweet.text,
                        "created_at": tweet.created_at,
                        "retweet_count": tweet.public_metrics["retweet_count"],
                        "reply_count": tweet.public_metrics["reply_count"],
                        "like_count": tweet.public_metrics["like_count"],
                        "quote_count": tweet.public_metrics["quote_count"],
                        "author_id": tweet.author_id,
                        "ticker": ticker
                    })
                    
        except Exception as e:
            logger.error(f"Error collecting tweets: {e}")
        
        return pd.DataFrame(tweets)

if __name__ == "__main__":
    collector = TwitterCollector()
    tweets_df = collector.collect_tweets("TSLA", max_results=50)
    print(f"Collected {len(tweets_df)} tweets")
    print(tweets_df.head())