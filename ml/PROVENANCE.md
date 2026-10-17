# Provenance

- Data: SMS Spam Collection v.1, Tiago A. Almeida and José María Gómez Hidalgo,
  UCI Machine Learning Repository (CC BY 4.0).
- Raw file -> `ml/data_pipeline.py` -> `data/sms.csv` (v1 / v2, tracked with DVC on Azure)
- `data/sms.csv` (v2) -> `ml/train.py --config ml/configs/char_logreg_v2.yaml` -> MLflow run
  `char-logreg-v2` -> `app/model/model.joblib` -> Docker image
