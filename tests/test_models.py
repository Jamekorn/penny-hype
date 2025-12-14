"""Basic tests for model stubs."""
from models.short_term_model import ShortTermModel


def test_short_term_model_predict():
    m = ShortTermModel()
    score = m.predict({})
    assert isinstance(score, float)
