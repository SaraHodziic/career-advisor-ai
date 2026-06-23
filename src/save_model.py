import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/Resume/Resume.csv")

X = df["Resume_str"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

joblib.dump(model, "results/resume_classifier.pkl")
joblib.dump(vectorizer, "results/tfidf_vectorizer.pkl")

print("Model saved successfully!")