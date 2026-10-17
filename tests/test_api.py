from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_spam():
    text = "WINNER!! You have won a 1000 pound prize. Call 09061701461 now to claim"
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert response.json()["label"] == "spam"


def test_predict_ham():
    response = client.post("/predict", json={"text": "Are we still meeting at 6 tonight?"})
    assert response.status_code == 200
    assert response.json()["label"] == "spam"


def test_probability_is_between_0_and_1():
    response = client.post("/predict", json={"text": "hello"})
    assert 0 <= response.json()["spam_probability"] <= 1


def test_empty_text_is_rejected():
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_blank_text_is_rejected():
    assert client.post("/predict", json={"text": " \n"}).status_code == 422


def test_too_long_text_is_rejected():
    assert client.post("/predict", json={"text": "a" * 1001}).status_code == 422


def test_missing_text_field():
    assert client.post("/predict", json={"message": "hi"}).status_code == 422


def test_text_must_be_a_string():
    assert client.post("/predict", json={"text": 12345}).status_code == 422


def test_emoji_and_accents_are_ok():
    response = client.post("/predict", json={"text": "Merci beaucoup 😀 à demain"})
    assert response.status_code == 200


def test_api_key_required_when_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    body = {"text": "see you tomorrow"}
    assert client.post("/predict", json=body).status_code == 401
    assert client.post("/predict", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.post("/predict", json=body, headers={"X-API-Key": "secret123"})
    assert ok.status_code == 200
