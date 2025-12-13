
## Directory Overview

### **data_collection/**
- `reddit_collector.py` - Collects data from Reddit discussions
- `twitter_collector.py` - Gathers tweets and social sentiment
- `stocktwits_collector.py` - Collects StockTwits sentiment data
- `sec_edgar_collector.py` - Fetches SEC/EDGAR filings
- `price_collector.py` - Retrieves historical stock prices

### **feature_engineering/**
- `hype_features.py` - Generates features from social media hype
- `reality_features.py` - Creates features from fundamental data
- `bot_detection.py` - Identifies and filters bot activity
- `feature_store.py` - Manages feature storage and retrieval

### **models/**
- `short_term_model.py` - Short-term prediction model
- `long_term_model.py` - Long-term prediction model
- `model_training.py` - Model training pipeline

### **deployment/**
- **api/** - REST API for model serving
  - `main.py` - FastAPI/FastAPI application
  - `endpoints.py` - API endpoint definitions
- **dashboard/** - Web dashboard interface
  - `app.py` - Streamlit/Dash application
  - `components.py` - UI components
- **orchestration/** - Scheduling and monitoring
  - `scheduler.py` - Task scheduling
  - `alert_service.py` - Alert notifications

### **config/**
- `settings.py` - Configuration settings
- `tickers.yaml` - Ticker symbols configuration

### **tests/**
- Unit tests for different modules

### **Root Files**
- `docker-compose.yml` - Docker Compose configuration
- `Dockerfile` - Container definition
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template
- `README.md` - Project documentation
