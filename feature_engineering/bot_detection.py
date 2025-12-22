import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import List, Dict
from loguru import logger
from transformers import pipeline

class BotDetector:
    def __init__(self):
        # Load AI text detector (simplified version)
        try:
            self.ai_detector = pipeline(
                "text-classification",
                model="roberta-base-openai-detector",
                device=-1  # Use CPU
            )
        except:
            self.ai_detector = None
            logger.warning("AI detector model not available, using rule-based approach")
    
    def detect_llm_generated_text(self, text: str) -> float:
        """Detect if text is AI-generated"""
        if not self.ai_detector or not isinstance(text, str) or len(text) < 10:
            return 0.0
        
        try:
            result = self.ai_detector(text[:512])  # Limit text length
            # Returns probability of being AI-generated
            if result[0]['label'] == 'AI':
                return result[0]['score']
            else:
                return 1 - result[0]['score']
        except:
            return 0.0
    
    def analyze_account_age(self, account_creation_dates: List[datetime], 
                          threshold_days: int = 30) -> Dict:
        """Analyze account age distribution"""
        if not account_creation_dates:
            return {"new_account_ratio": 0.0, "median_account_age_days": 0}
        
        now = datetime.now()
        ages_days = []
        new_accounts = 0
        
        for date in account_creation_dates:
            if date:
                age_days = (now - date).days
                ages_days.append(age_days)
                if age_days <= threshold_days:
                    new_accounts += 1
        
        features = {
            "new_account_ratio": new_accounts / len(ages_days) if ages_days else 0,
            "median_account_age_days": np.median(ages_days) if ages_days else 0,
            "min_account_age_days": min(ages_days) if ages_days else 0,
            "accounts_analyzed": len(ages_days)
        }
        
        return features
    
    def detect_repeated_messages(self, messages: List[str], 
                               similarity_threshold: float = 0.8) -> Dict:
        """Detect repeated or similar messages"""
        if not messages or len(messages) < 2:
            return {"repetition_ratio": 0.0, "unique_messages": len(messages)}
        
        # Simple text similarity using Jaccard similarity
        def jaccard_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0
        
        # Find similar messages
        similar_pairs = 0
        total_pairs = 0
        
        for i in range(len(messages)):
            for j in range(i + 1, len(messages)):
                if isinstance(messages[i], str) and isinstance(messages[j], str):
                    similarity = jaccard_similarity(messages[i], messages[j])
                    if similarity > similarity_threshold:
                        similar_pairs += 1
                    total_pairs += 1
        
        # Analyze message lengths
        message_lengths = [len(str(msg)) for msg in messages if isinstance(msg, str)]
        
        features = {
            "repetition_ratio": similar_pairs / total_pairs if total_pairs > 0 else 0,
            "unique_messages": len(set(messages)),
            "avg_message_length": np.mean(message_lengths) if message_lengths else 0,
            "short_message_ratio": sum(1 for l in message_lengths if l < 20) / len(message_lengths) if message_lengths else 0
        }
        
        return features
    
    def detect_bot_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect various bot patterns"""
        features = {}
        
        if df.empty:
            return features
        
        # 1. Posting frequency analysis
        if 'created_utc' in df.columns and 'author' in df.columns:
            df['created_utc'] = pd.to_datetime(df['created_utc'])
            df = df.sort_values('created_utc')
            
            # Calculate posting frequency per author
            author_freq = df.groupby('author').size()
            features["authors_total"] = len(author_freq)
            features["posts_per_author_avg"] = author_freq.mean() if not author_freq.empty else 0
            features["posts_per_author_std"] = author_freq.std() if not author_freq.empty else 0
            
            # Flag suspicious frequency
            if not author_freq.empty:
                suspicious_authors = author_freq[author_freq > 10]  # More than 10 posts
                features["high_frequency_authors"] = len(suspicious_authors)
                features["high_frequency_ratio"] = len(suspicious_authors) / len(author_freq)
        
        # 2. Content analysis for bot-like patterns
        if 'content' in df.columns:
            messages = df['content'].dropna().tolist()
            
            # Check for AI-generated content
            ai_scores = []
            for msg in messages[:10]:  # Sample first 10 for speed
                ai_score = self.detect_llm_generated_text(msg[:500])
                ai_scores.append(ai_score)
            
            features["avg_ai_score"] = np.mean(ai_scores) if ai_scores else 0
            features["high_ai_score_ratio"] = sum(1 for s in ai_scores if s > 0.7) / len(ai_scores) if ai_scores else 0
            
            # Check for repeated messages
            repetition_features = self.detect_repeated_messages(messages)
            features.update(repetition_features)
        
        # 3. Temporal patterns
        if 'created_utc' in df.columns:
            df['hour'] = df['created_utc'].dt.hour
            hourly_dist = df['hour'].value_counts().sort_index()
            
            # Check for unnatural posting times (e.g., all at same hour)
            if not hourly_dist.empty:
                features["temporal_concentration"] = hourly_dist.max() / hourly_dist.sum()
                
                # Flag if most posts in a short time window
                time_diff = df['created_utc'].diff().dropna()
                if not time_diff.empty:
                    rapid_posts = sum(time_diff < pd.Timedelta(seconds=10))
                    features["rapid_post_ratio"] = rapid_posts / len(df)
        
        # 4. Combined bot probability score
        bot_score = 0.0
        
        # Weight different indicators
        if "high_frequency_ratio" in features:
            bot_score += features["high_frequency_ratio"] * 0.3
        
        if "avg_ai_score" in features:
            bot_score += features["avg_ai_score"] * 0.3
        
        if "repetition_ratio" in features:
            bot_score += features["repetition_ratio"] * 0.2
        
        if "rapid_post_ratio" in features:
            bot_score += features["rapid_post_ratio"] * 0.2
        
        features["bot_probability"] = min(bot_score, 1.0)
        
        # Classification
        if features["bot_probability"] > 0.7:
            features["bot_likelihood"] = "high"
        elif features["bot_probability"] > 0.4:
            features["bot_likelihood"] = "medium"
        else:
            features["bot_likelihood"] = "low"
        
        return features