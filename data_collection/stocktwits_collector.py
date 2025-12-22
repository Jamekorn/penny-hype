"""StockTwits collector stub."""
from typing import List, Dict


def collect_stocktwits_messages(ticker: str, limit: int = 100) -> List[Dict]:
    return [{"id": "stub", "ticker": ticker, "text": "sample stocktwits message"}]
