import os
import joblib
import uvicorn

from data import data
from ml import model
from ml import model_features

class Config:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "logreg")
        self.model_path = os.getenv("MODEL_PATH", "./data/train/model.pkl")
        self.model_balanced = os.getenv("MODEL_BALANCED", 1)
        self.model_dataset_db_path = os.getenv("MODEL_DATASET_DB_PATH", "./data/dataset/sqlite/spamclassification.db")
        self.model_dataset_raw_path = os.getenv("MODEL_DATASET_RAW_PATH", "./data/dataset/raw/dataset_spam.csv")
        self.fastapi_port = int(os.getenv("BACKEND_PORT", 8080))

if __name__ == "__main__":
    config = Config()
    if not os.path.exists(config.model_path):
        storage = data.Storage(config.model_dataset_raw_path, config.model_dataset_db_path, "spamclassification")
        storage.init()

        df = storage.load_data()
        X = df[model_features.FEATURES]
        y = df[model_features.TARGET]

        build_model = model.build_model(config.model_name, config.model_balanced)

        print(f"Model - {config.model_name}")
        print(f"Model balanced - {True if config.model_balanced else False}")
        print(build_model.benchmark_model(X, y))

        build_model.train_model(X, y)
        print("[+] Train model")

        joblib.dump({
            "model": build_model.model,
            "model_name": config.model_name,
            "vectorizer": build_model.vectorizer,
            "features": model_features.FEATURES,
            "target": model_features.TARGET
        }, config.model_path)

    uvicorn.run("api.api:app", host="0.0.0.0", port=config.fastapi_port)