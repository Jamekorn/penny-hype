import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # Data Sources
    TICKERS = ["AMC", "GME", "BBBY", "TSLA", "AAPL"]  # Example tickers
    
    # Time Windows
    SHORT_TERM_WINDOW = int(os.getenv("SHORT_TERM_DAYS", 1))
    LONG_TERM_WINDOW = int(os.getenv("LONG_TERM_DAYS", 7))
    
    # API Credentials
    REDDIT_CONFIG = {
        "client_id": os.getenv("REDDIT_CLIENT_ID"),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
        "user_agent": os.getenv("REDDIT_USER_AGENT", "stock_analyzer/1.0")
    }
    
    TWITTER_CONFIG = {
        "bearer_token": os.getenv("TWITTER_BEARER_TOKEN"),
        "api_key": os.getenv("TWITTER_API_KEY"),
        "api_secret": os.getenv("TWITTER_API_SECRET")
    }
    
    # Database
    DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    
    # Redis
    REDIS_URL = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
    
    # Model Thresholds
    THRESHOLD_BUY = float(os.getenv("THRESHOLD_BUY", 0.6))
    THRESHOLD_SELL = float(os.getenv("THRESHOLD_SELL", 0.4))

settings = Settings()