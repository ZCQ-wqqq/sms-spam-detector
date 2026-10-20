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


## Project structure

```
app/            FastAPI app, web page and the trained model (app/model/)
ml/             data pipeline, training script, configs and ML docs
data/           DVC pointer files (the real data is on Azure)
tests/          pytest tests
.github/        CI and CD workflows
Dockerfile      image for running the app
```

## Branches

We use 4 kinds of branches:

- `feature/...` one branch per task, created from `dev`
- `dev` where features are merged
- `staging` tested version before production
- `main` production

Code goes `feature/...` -> `dev` -> `staging` -> `main`, always with a pull request.
`dev`, `staging` and `main` are protected: no direct push, 1 approval needed and the CI
must pass before merging.

## From commit to production

1. Commit on a feature branch and open a PR to `dev`.
2. CI runs on the PR: ruff (lint + format), pytest, docker build + a quick test of the
   container. If one job fails the PR can't be merged.
3. A teammate reviews and approves, then we merge into `dev`.
4. PR `dev` -> `staging`: CI runs again. After the merge, the CD workflow creates a
   staging `.env` from the staging secrets and builds the staging image.
5. PR `staging` -> `main`: CI runs again. After the merge, the CD workflow does the same
   with the production secrets and builds the production image.

| Event | Workflow | What runs |
|---|---|---|
| PR to dev / staging / main | `ci.yml` | lint, test, docker |
| merge (push) to staging | `cd.yml` | staging job |
| merge (push) to main | `cd.yml` | production job |

More details about the secrets in [DEPLOYMENT.md](DEPLOYMENT.md).
The data and the experiments are explained in [ml/README_data.md](ml/README_data.md) and
[ml/README_experiments.md](ml/README_experiments.md).

## Maintained by

- Zhang Jingyi — app & tests
- Zhao Tong — data pipeline & model training
- Zhang Chaoqian — docker, CI/CD & deployment