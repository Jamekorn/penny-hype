import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.model_training import ModelTrainer
import yfinance as yf

# Download sample data
tickers = ["GME", "AMC", "TSLA", "AAPL", "NVDA"]
all_data = []

for ticker in tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    hist['ticker'] = ticker
    all_data.append(hist)

# Combine data
price_data = pd.concat(all_data)

# Create sample features (in reality, you'd use your feature engineering)
# This is just for demonstration
sample_features = {
    'attention_velocity': np.random.randn(100),
    'sentiment_mean': np.random.randn(100),
    'gini_coefficient': np.random.rand(100),
    'dilution_risk': np.random.rand(100),
    'cash_runway': np.random.rand(100) * 24,
    'bot_probability': np.random.rand(100),
    'volume_zscore': np.random.randn(100)
}

features_df = pd.DataFrame(sample_features)

# Create labels (random for demo)
features_df['target'] = np.random.randint(0, 2, 100)

# Train models
trainer = ModelTrainer()

print("Training short-term model...")
short_term_metrics = trainer.train_short_term_model(features_df)

print("Training long-term model...")
long_term_metrics = trainer.train_long_term_model(features_df)

# Save models
trainer.save_models()

print("Models trained and saved successfully!")