import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy import stats
from loguru import logger

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

class HypeFeatureEngineer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def calculate_attention_velocity(self, df: pd.DataFrame, 
                                   time_col: str = 'created_utc',
                                   window_hours: int = 24) -> pd.Series:
        """Calculate Z-score of message count over time"""
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        
        # Resample to hourly counts
        hourly_counts = df.resample('H').size()
        
        # Calculate rolling mean and std
        rolling_mean = hourly_counts.rolling(f'{window_hours}H').mean()
        rolling_std = hourly_counts.rolling(f'{window_hours}H').std()
        
        # Calculate Z-score
        z_scores = (hourly_counts - rolling_mean) / rolling_std
        z_scores = z_scores.fillna(0)
        
        return z_scores
    
    def calculate_sentiment_features(self, texts: List[str]) -> Dict:
        """Calculate sentiment features using VADER"""
        sentiments = []
        for text in texts:
            if isinstance(text, str):
                score = self.sia.polarity_scores(text)
                sentiments.append(score)
            else:
                sentiments.append({'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})
        
        df_sentiments = pd.DataFrame(sentiments)
        
        features = {
            'mean_compound': df_sentiments['compound'].mean(),
            'std_compound': df_sentiments['compound'].std(),
            'mean_positive': df_sentiments['pos'].mean(),
            'mean_negative': df_sentiments['neg'].mean(),
            'pos_neg_ratio': df_sentiments['pos'].mean() / max(df_sentiments['neg'].mean(), 0.001),
            'sentiment_volatility': df_sentiments['compound'].std(),
            'extreme_sentiment_pct': (abs(df_sentiments['compound']) > 0.5).mean()
        }
        
        return features
    
    def calculate_concentration_gini(self, df: pd.DataFrame, 
                                    author_col: str = 'author') -> float:
        """Calculate Gini coefficient for author concentration"""
        if author_col not in df.columns or df[author_col].isnull().all():
            return 0.0
        
        # Count posts per author
        author_counts = df[author_col].value_counts()
        
        # Sort counts
        counts_sorted = np.sort(author_counts.values)
        n = len(counts_sorted)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(counts_sorted)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return gini
    
    def calculate_cross_platform_correlation(self, platform_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate correlation between platforms"""
        correlations = {}
        platforms = list(platform_data.keys())
        
        for i, platform1 in enumerate(platforms):
            for platform2 in platforms[i+1:]:
                df1 = platform_data[platform1]
                df2 = platform_data[platform2]
                
                # Align timestamps
                df1['hour'] = pd.to_datetime(df1['created_utc']).dt.floor('H')
                df2['hour'] = pd.to_datetime(df2['created_utc']).dt.floor('H')
                
                hourly1 = df1.groupby('hour').size()
                hourly2 = df2.groupby('hour').size()
                
                # Merge
                merged = pd.concat([hourly1, hourly2], axis=1, keys=[platform1, platform2])
                merged = merged.fillna(0)
                
                if len(merged) > 1:
                    corr = merged[platform1].corr(merged[platform2])
                    correlations[f"{platform1}_{platform2}_volume_corr"] = corr
        
        return correlations
    
    def extract_all_hype_features(self, 
                                 reddit_data: pd.DataFrame,
                                 twitter_data: pd.DataFrame,
                                 stocktwits_data: pd.DataFrame) -> Dict:
        """Extract all hype tower features"""
        features = {}
        
        # Attention Velocity for each platform
        for platform_name, data in [
            ('reddit', reddit_data),
            ('twitter', twitter_data),
            ('stocktwits', stocktwits_data)
        ]:
            if len(data) > 0:
                z_scores = self.calculate_attention_velocity(data)
                features[f'{platform_name}_attention_zscore'] = z_scores.iloc[-1] if len(z_scores) > 0 else 0
                features[f'{platform_name}_volume'] = len(data)
        
        # Sentiment Analysis
        all_texts = []
        platform_texts = {}
        
        for platform_name, data in [
            ('reddit', reddit_data),
            ('twitter', twitter_data),
            ('stocktwits', stocktwits_data)
        ]:
            if len(data) > 0:
                texts = []
                if 'content' in data.columns:
                    texts = data['content'].dropna().tolist()
                elif 'title' in data.columns:
                    texts = data['title'].dropna().tolist()
                
                if texts:
                    platform_texts[platform_name] = texts
                    all_texts.extend(texts)
        
        # Overall sentiment
        if all_texts:
            sentiment_features = self.calculate_sentiment_features(all_texts)
            for key, value in sentiment_features.items():
                features[f'overall_{key}'] = value
        
        # Platform-specific sentiment
        for platform_name, texts in platform_texts.items():
            if texts:
                platform_sentiment = self.calculate_sentiment_features(texts)
                for key, value in platform_sentiment.items():
                    features[f'{platform_name}_{key}'] = value
        
        # Concentration (Gini)
        for platform_name, data in [
            ('reddit', reddit_data),
            ('twitter', twitter_data),
            ('stocktwits', stocktwits_data)
        ]:
            if len(data) > 0 and 'author' in data.columns:
                gini = self.calculate_concentration_gini(data)
                features[f'{platform_name}_gini'] = gini
        
        # Cross-platform correlation
        platform_data = {}
        for platform_name, data in [
            ('reddit', reddit_data),
            ('twitter', twitter_data),
            ('stocktwits', stocktwits_data)
        ]:
            if len(data) > 0:
                platform_data[platform_name] = data
        
        if len(platform_data) >= 2:
            correlations = self.calculate_cross_platform_correlation(platform_data)
            features.update(correlations)
        
        return features