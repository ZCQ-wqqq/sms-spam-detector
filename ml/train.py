import argparse
import hashlib
import json
import os
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA_PATH = "data/sms.csv"


def build_model(cfg):
    if cfg["model"] == "naive_bayes":
        return make_pipeline(CountVectorizer(), MultinomialNB(alpha=cfg["alpha"]))

    if cfg["model"] == "char_logreg":
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        classifier = LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=2000)
        return make_pipeline(vectorizer, classifier)

    raise ValueError(f"unknown model: {cfg['model']}")


def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def dvc_md5(path):
    # md5 written by "dvc add" in the .dvc pointer file
    pointer = path + ".dvc"
    if not os.path.exists(pointer):
        return "not tracked"
    with open(pointer) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def save_confusion_matrix(y_test, pred, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["ham", "spam"])
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--export", help="folder where the model for the app is saved")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(DATA_PATH)
    y = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(cfg)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        not_params = ("experiment", "run_name", "register_as")
        mlflow.log_params({k: v for k, v in cfg.items() if k not in not_params})
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        mlflow.set_tag("git_sha", git_sha())
        mlflow.set_tag("data_md5", file_md5(DATA_PATH))
        mlflow.set_tag("dvc_md5", dvc_md5(DATA_PATH))

        mlflow.log_metrics(metrics)

        save_confusion_matrix(y_test, pred, "confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        os.remove("confusion_matrix.png")

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=cfg.get("register_as"),
        )

        run_id = run.info.run_id

    print(f"run {cfg['run_name']} ({run_id})")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    if args.export:
        os.makedirs(args.export, exist_ok=True)
        joblib.dump(model, os.path.join(args.export, "model.joblib"))

        info = {
            "run_id": run_id,
            "run_name": cfg["run_name"],
            "data_version": cfg["data_version"],
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
        }
        with open(os.path.join(args.export, "model_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"model saved to {args.export}")


if __name__ == "__main__":
    main()
