# Current production model

- MLflow registered model: `sms-spam-model`, version 1
- Run: `char-logreg-v2` (run id in `app/model/model_info.json`)
- Data: v2 (tag `data-v2`)
- Model: TF-IDF on character n-grams (2 to 5) + Logistic Regression (C=10, balanced)
- File used by the app: `app/model/model.joblib`

To change the production model: train with `--export app/model`, update this file and
`README_experiments.md`, then open a PR.
