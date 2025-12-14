from fastapi import APIRouter, HTTPException
from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd
import json

router = APIRouter()

# Mock endpoints for additional functionality

@router.get("/historical/{ticker}")
async def get_historical_predictions(ticker: str, days: int = 7):
    """Get historical predictions for a ticker"""
    # This would query a database in production
    return {
        "ticker": ticker,
        "days": days,
        "predictions": [
            {
                "date": (datetime.now() - timedelta(days=i)).date().isoformat(),
                "prediction": "BUY" if i % 3 == 0 else "SELL",
                "confidence": 0.7 - (i * 0.05)
            }
            for i in range(days)
        ]
    }

@router.get("/leaderboard")
async def get_leaderboard(metric: str = "accuracy", limit: int = 10):
    """Get leaderboard of best performing predictions"""
    # Mock data
    return {
        "metric": metric,
        "leaderboard": [
            {
                "ticker": f"TICK{i+1}",
                metric: 0.9 - (i * 0.05),
                "total_predictions": 100 - i,
                "success_rate": 0.85 - (i * 0.03)
            }
            for i in range(limit)
        ]
    }

@router.post("/feedback")
async def submit_feedback(
    ticker: str,
    prediction_correct: bool,
    actual_price_change: float,
    comments: str = None
):
    """Submit feedback on prediction accuracy"""
    # In production, this would save to a database
    return {
        "status": "feedback_received",
        "ticker": ticker,
        "prediction_correct": prediction_correct,
        "feedback_id": f"FB{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "received_at": datetime.now().isoformat()
    }

@router.get("/system_metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    return {
        "uptime": "99.5%",
        "average_response_time": "0.45s",
        "total_predictions": 12457,
        "accuracy_last_24h": 0.68,
        "active_models": 2,
        "data_sources": {
            "reddit": "active",
            "twitter": "active",
            "sec": "active",
            "stocktwits": "inactive"
        }
    }