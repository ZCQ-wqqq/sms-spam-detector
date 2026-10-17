import json

import joblib
import pytest

MODEL_PATH = "app/model/model.joblib"
INFO_PATH = "app/model/model_info.json"


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


def test_model_info_has_metrics():
    with open(INFO_PATH) as f:
        info = json.load(f)
    assert info["data_version"] == "v2"
    assert 0 <= info["metrics"]["f1"] <= 1


def test_obvious_spam(model):
    texts = [
        "URGENT! You have won a free cruise. Text WIN to 80086 now",
        "Congratulations, you are selected for a 2000 cash prize, call 0800 now",
    ]
    assert list(model.predict(texts)) == [1, 1]


def test_obvious_ham(model):
    texts = ["I will be home late, keep some dinner for me", "Ok see you at the library"]
    assert list(model.predict(texts)) == [0, 0]


def test_probabilities_sum_to_one(model):
    probs = model.predict_proba(["call me later", "free entry to win cash"])
    for row in probs:
        assert row.sum() == pytest.approx(1.0)


def test_uppercase_does_not_change_the_result(model):
    text = "free entry in a weekly competition to win cup final tickets"
    assert model.predict([text])[0] == model.predict([text.upper()])[0]


def test_single_character_message(model):
    assert model.predict(["k"])[0] in (0, 1)
