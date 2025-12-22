"""Price collector stub.

Use `yfinance`, `pandas-datareader` or exchange APIs to fetch historical prices.
"""
from typing import List, Dict
import datetime


def fetch_price_history(ticker: str, start: datetime.date, end: datetime.date) -> List[Dict]:
    """Return list of date/price dicts as a stub."""
    return [{"date": start.isoformat(), "close": 100.0}, {"date": end.isoformat(), "close": 101.0}]
