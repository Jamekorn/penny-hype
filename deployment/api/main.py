from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import redis
import json
from loguru import logger

from models.short_term_model import StockPredictionModel
from feature_engineering.hype_features import HypeFeatureEngineer
from feature_engineering.reality_features import RealityFeatureEngineer
from feature_engineering.bot_detection import BotDetector
from data_collection.reddit_collector import RedditCollector
from data_collection.twitter_collector import TwitterCollector
from data_collection.sec_edgar_collector import SECEdgarCollector

app = FastAPI(
    title="Stock Prediction API",
    description="Hybrid framework for predicting micro-cap stock movement",
    version="1.0.0"
)

# Initialize components
hype_engineer = HypeFeatureEngineer()
reality_engineer = RealityFeatureEngineer()
bot_detector = BotDetector()

# Initialize data collectors
reddit_collector = RedditCollector()
twitter_collector = TwitterCollector()
sec_collector = SECEdgarCollector()

# Initialize models (to be loaded from saved files)
short_term_model = None
long_term_model = None

# Initialize Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class PredictionRequest(BaseModel):
    ticker: str
    include_social: bool = True
    include_fundamentals: bool = True
    timeframe: str = "short"  # "short" or "long"

class PredictionResponse(BaseModel):
    ticker: str
    prediction: str  # "BUY", "SELL", "HOLD"
    confidence: float
    features: Dict
    timestamp: datetime
    model_type: str

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global short_term_model, long_term_model
    
    try:
        short_term_model = StockPredictionModel("xgboost")
        short_term_model.load("models/saved_models/short_term_model.pkl")
        logger.info("Short-term model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load short-term model: {e}")
        short_term_model = None
    
    try:
        long_term_model = StockPredictionModel("lightgbm")
        long_term_model.load("models/saved_models/long_term_model.pkl")
        logger.info("Long-term model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load long-term model: {e}")
        long_term_model = None

@app.get("/")
async def root():
    return {
        "message": "Stock Prediction API",
        "endpoints": {
            "/predict": "POST - Get prediction for a ticker",
            "/health": "GET - API health check",
            "/features/{ticker}": "GET - Get features for a ticker"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    model_status = {
        "short_term_model_loaded": short_term_model is not None,
        "long_term_model_loaded": long_term_model is not None,
        "redis_connected": redis_client.ping()
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_status": model_status
    }

async def collect_data(ticker: str) -> Dict:
    """Collect data from all sources"""
    data = {}
    
    # Check cache first
    cache_key = f"data:{ticker}:{datetime.now().strftime('%Y-%m-%d')}"
    cached_data = redis_client.get(cache_key)
    
    if cached_data:
        logger.info(f"Using cached data for {ticker}")
        return json.loads(cached_data)
    
    # Collect data in parallel
    tasks = []
    
    # Reddit data
    async def get_reddit_data():
        try:
            return await asyncio.to_thread(
                reddit_collector.collect_posts, ticker, limit_per_sub=50
            )
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
            return pd.DataFrame()
    
    # Twitter data
    async def get_twitter_data():
        try:
            return await asyncio.to_thread(
                twitter_collector.collect_tweets, ticker, max_results=100
            )
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {e}")
            return pd.DataFrame()
    
    # SEC data
    async def get_sec_data():
        try:
            return await asyncio.to_thread(
                sec_collector.get_filings, ticker, "10-K", limit=3
            )
        except Exception as e:
            logger.error(f"Error collecting SEC data: {e}")
            return pd.DataFrame()
    
    # Execute all collection tasks
    reddit_task = asyncio.create_task(get_reddit_data())
    twitter_task = asyncio.create_task(get_twitter_data())
    sec_task = asyncio.create_task(get_sec_data())
    
    reddit_data, twitter_data, sec_data = await asyncio.gather(
        reddit_task, twitter_task, sec_task
    )
    
    data = {
        "reddit": reddit_data.to_dict(orient='records') if not reddit_data.empty else [],
        "twitter": twitter_data.to_dict(orient='records') if not twitter_data.empty else [],
        "sec": sec_data.to_dict(orient='records') if not sec_data.empty else [],
        "collected_at": datetime.now().isoformat()
    }
    
    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(data, default=str))
    
    return data

@app.get("/features/{ticker}")
async def get_features(ticker: str):
    """Get engineered features for a ticker"""
    try:
        # Collect data
        raw_data = await collect_data(ticker)
        
        # Convert back to DataFrames
        reddit_df = pd.DataFrame(raw_data["reddit"]) if raw_data["reddit"] else pd.DataFrame()
        twitter_df = pd.DataFrame(raw_data["twitter"]) if raw_data["twitter"] else pd.DataFrame()
        sec_df = pd.DataFrame(raw_data["sec"]) if raw_data["sec"] else pd.DataFrame()
        
        # Engineer features
        hype_features = hype_engineer.extract_all_hype_features(
            reddit_df, twitter_df, pd.DataFrame()  # Add StockTwits if available
        )
        
        reality_features = reality_engineer.extract_all_reality_features(
            sec_df, pd.DataFrame()  # Add financial data if available
        )
        
        bot_features = bot_detector.detect_bot_patterns(
            pd.concat([reddit_df, twitter_df], ignore_index=True)
        )
        
        # Combine all features
        all_features = {**hype_features, **reality_features, **bot_features}
        
        return {
            "ticker": ticker,
            "features": all_features,
            "feature_count": len(all_features),
            "hype_features_count": len(hype_features),
            "reality_features_count": len(reality_features),
            "bot_features_count": len(bot_features)
        }
        
    except Exception as e:
        logger.error(f"Error getting features for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Get prediction for a stock"""
    try:
        # Validate timeframe
        if request.timeframe not in ["short", "long"]:
            raise HTTPException(
                status_code=400, 
                detail="timeframe must be 'short' or 'long'"
            )
        
        # Get features
        features_response = await get_features(request.ticker)
        features = features_response["features"]
        
        # Convert features to array
        feature_values = list(features.values())
        feature_array = np.array([feature_values])
        
        # Select model based on timeframe
        if request.timeframe == "short":
            if short_term_model is None:
                raise HTTPException(
                    status_code=503,
                    detail="Short-term model not available"
                )
            
            model = short_term_model
            model_type = "short_term"
        else:
            if long_term_model is None:
                raise HTTPException(
                    status_code=503,
                    detail="Long-term model not available"
                )
            
            model = long_term_model
            model_type = "long_term"
        
        # Make prediction
        prediction, probabilities = model.predict(feature_array)
        
        # Convert prediction to action
        pred_value = int(prediction[0])
        confidence = float(probabilities[0][pred_value])
        
        if pred_value == 1:
            action = "BUY"
        else:
            action = "SELL"
        
        # Adjust confidence based on feature quality
        if len(features) < 10:
            confidence *= 0.7  # Penalize for few features
        
        return PredictionResponse(
            ticker=request.ticker,
            prediction=action,
            confidence=confidence,
            features=features,
            timestamp=datetime.now(),
            model_type=model_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/batch_predict")
async def batch_predict(tickers: str):
    """Get predictions for multiple tickers"""
    ticker_list = [t.strip() for t in tickers.split(",")]
    
    results = []
    for ticker in ticker_list:
        try:
            prediction = await predict(PredictionRequest(
                ticker=ticker,
                timeframe="short"
            ))
            results.append(prediction.dict())
        except Exception as e:
            results.append({
                "ticker": ticker,
                "error": str(e)
            })
    
    return {
        "results": results,
        "total": len(results),
        "successful": sum(1 for r in results if "error" not in r)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )