"""Simple feature store wrapper (in-memory stub)."""
from typing import Dict, Any


class FeatureStore:
    def __init__(self):
        self._store = {}

    def upsert(self, key: str, features: Dict[str, Any]):
        self._store[key] = features

    def get(self, key: str):
        return self._store.get(key)
