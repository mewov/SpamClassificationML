from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

class Model():
    def __init__(self, model: LogisticRegression | SVC | LinearSVC):
        self.model = model
        self.vectorizer = TfidfVectorizer()

    def benchmark_model(self, X: any, y: any):
        pipeline = make_pipeline(TfidfVectorizer(), clone(self.model))
        y_pred = cross_val_predict(pipeline, X, y, cv=5)
        return classification_report(y, y_pred)
    
    def train_model(self, X: any, y: any):
        X = self.vectorizer.fit_transform(X)
        self.model.fit(X, y)
        
    def predict(self, X: any):
        X = self.vectorizer.transform(X)
        return self.model.predict(X)[0]

def build_model(model_name: str, model_balanced: int):
    if model_name == "logreg": return Model(LogisticRegression(class_weight="balanced" if model_balanced else None))
    elif model_name == "lsvc": return Model(LinearSVC(class_weight="balanced" if model_balanced else None))
    elif model_name == "svc": return Model(SVC(class_weight="balanced" if model_balanced else None))
    else: raise ValueError("Invalid model")
