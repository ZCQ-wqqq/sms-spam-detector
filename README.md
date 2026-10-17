# SMS Spam Detector
Final project for the DevOps & MLOps course.
A small web app that tells if an SMS message is spam or not. You type a message,
the app sends it to a model we trained with scikit-learn and shows the result.
Team: Zhang Chaoqian, Zhang Jingyi, Zhao Tong

## Run it locally

```bash
python -m venv .venv
# Windows Git Bash
export PATH="$PWD/.venv/Scripts:$PATH"
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000

Run the tests and the linter:

```bash
pytest
ruff check .
ruff format --check .
```

## API

- `GET /` web page
- `GET /health` returns `{"status": "ok", "env": ...}`
- `GET /model-info` which model is used and its metrics
- `POST /predict` with `{"text": "your message"}`, returns the label (`spam` / `ham`)
  and the spam probability. When `API_KEY` is set (staging and production) the request needs the header `X-API-Key`.
