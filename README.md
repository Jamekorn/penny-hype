General Framework
stock_prediction_framework/
├── data_collection/
│   ├── __init__.py
│   ├── reddit_collector.py
│   ├── twitter_collector.py
│   ├── stocktwits_collector.py
│   ├── sec_edgar_collector.py
│   └── price_collector.py
├── feature_engineering/
│   ├── __init__.py
│   ├── hype_features.py
│   ├── reality_features.py
│   ├── bot_detection.py
│   └── feature_store.py
├── models/
│   ├── __init__.py
│   ├── short_term_model.py
│   ├── long_term_model.py
│   └── model_training.py
├── deployment/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── endpoints.py
│   ├── dashboard/
│   │   ├── app.py
│   │   └── components.py
│   └── orchestration/
│       ├── scheduler.py
│       └── alert_service.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── tickers.yaml
├── tests/
│   ├── test_data_collection.py
│   ├── test_features.py
│   └── test_models.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
