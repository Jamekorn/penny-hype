import pytest
import pandas as pd
from datetime import datetime
from data_collection.reddit_collector import RedditCollector
from data_collection.twitter_collector import TwitterCollector

class TestDataCollection:
    def test_reddit_collector_initialization(self):
        """Test Reddit collector initialization"""
        collector = RedditCollector()
        assert collector.reddit is not None
    
    def test_twitter_collector_initialization(self):
        """Test Twitter collector initialization"""
        collector = TwitterCollector()
        assert collector.client is not None
    
    @pytest.mark.skip("Requires API keys")
    def test_reddit_data_collection(self):
        """Test Reddit data collection"""
        collector = RedditCollector()
        data = collector.collect_posts("GME", limit_per_sub=2)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'title' in data.columns
    
    @pytest.mark.skip("Requires API keys")
    def test_twitter_data_collection(self):
        """Test Twitter data collection"""
        collector = TwitterCollector()
        data = collector.collect_tweets("TSLA", max_results=5)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'content' in data.columns