# Spam Classification

Small ML project for spam message classification with model training, saved artifacts, and a FastAPI endpoint for inference.

## What's Inside

- `LogisticRegression`, `LinearSVC`, `SVC`
- classification benchmarking via cross-validation (`classification_report`)
- model saving to `data/train/model.pkl`
- FastAPI prediction endpoint
- SQLite dataset cache built from CSV
- confusion matrix images in `assets/`

## Features

Input feature:

- `message`

Target:

- `class` (`0` = not spam, `1` = spam)

Text vectorization:

- `TfidfVectorizer`

## Install

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
python app_main.py
```

What happens on startup:

- if `data/train/model.pkl` does not exist, the project loads the dataset, benchmarks the selected model, trains it, and saves it;
- then FastAPI starts on port `8080` by default.

## Environment Variables

Example config:

```env
MODEL_NAME=logreg
MODEL_PATH=./data/train/model.pkl
MODEL_BALANCED=1
MODEL_DATASET_DB_PATH=./data/dataset/sqlite/spamclassification.db
MODEL_DATASET_RAW_PATH=./data/dataset/raw/dataset_spam.csv
BACKEND_PORT=8080
```

Available model names:

- `logreg`
- `lsvc`
- `svc`

## API

After startup, the service is available at:

```text
http://127.0.0.1:8080
```

Prediction endpoint:

```text
POST /predict
```

Request example:

```json
{
  "message": "Congratulations! You won a free gift card. Click now."
}
```

Response example:

```json
{
  "is_ok": true,
  "is_spam": true
}
```

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

## Assets

The `assets/` folder stores experiment visuals.

Available plots:

- `assets/confusion_matrix_logres.png`
- `assets/confusion_matrix_lsvc.png`
- `assets/confusion_matrix_svc.png`

## Dataset Format

Expected source columns:

```text
message,class
```

If you replace the dataset with your own data in the same format, retrain the model and the API will use the new artifact.

## Notes

- quality depends on dataset distribution and text quality;
- the model is lightweight and easy to retrain;
- TF-IDF is used as a simple and strong baseline for short text classification.

## Limitations

- baseline text-classification setup without advanced NLP features;
- no probability calibration in current API response;
- quality may drop on data very different from the training distribution.
