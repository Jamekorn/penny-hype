"""Basic tests for feature functions."""
from feature_engineering.hype_features import compute_hype_features


def test_hype_features_basic():
    res = compute_hype_features([{"text": "hi"}])
    assert "mention_count" in res
