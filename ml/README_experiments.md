# Experiments

Training script: `ml/train.py`, configs in `ml/configs/`.

All runs are logged in MLflow (local, `mlflow.db`). For each run we log the git commit
(`git_sha`), the data md5 and DVC md5, the parameters, the metrics, the confusion matrix
and the model.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Metric

We look mostly at the **F1 score of the spam class**. Only 13% of the messages are spam,
so a model that always says "ham" already gets 87% accuracy and is useless. F1 combines
precision (when we say spam, is it really spam?) and recall (how many spam messages do
we catch?).

## Runs

Same split for all runs: 80% train / 20% test, stratified, `random_state=42`.

| Experiment | Run | Data | Model | Accuracy | Precision | Recall | F1 | Run ID |
|---|---|---|---|---|---|---|---|---|
| sms-baseline | nb-v1 | v1 | CountVectorizer + Naive Bayes | 0.9848 | 0.9783 | 0.9060 | 0.9408 | `b09687e243e6402aba6412a830f165ce` |
| sms-baseline | nb-v2 | v2 | CountVectorizer + Naive Bayes | 0.9845 | 0.9667 | 0.9062 | 0.9355 | `868334c8b3844497bd7d4deefafc4e4c` |
| sms-improved | char-logreg-v2 | v2 | char n-gram TF-IDF + Logistic Regression | 0.9932 | 0.9840 | 0.9609 | 0.9723 | `3f67fed0698a49489738e3d33ad8d18a` |

## What we learned

- nb-v1 -> nb-v2: same model, only the data changed. The score goes a little down because
  v1 had duplicates in train and test at the same time (leakage). The v2 score is the
  real one.
- nb-v2 -> char-logreg-v2: same data, only the model changed. Character n-grams also
  catch things like "fr33", "£", "www", phone numbers and spelling mistakes, and
  `class_weight="balanced"` helps with the small number of spam messages.

## Production model

We use **char-logreg-v2** because it has the best F1 on the same data as the baseline
(v2), with higher precision and recall. It is registered in MLflow as `sms-spam-model`
and exported to `app/model/`. See [registry/current_model.md](registry/current_model.md).
