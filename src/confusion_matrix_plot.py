import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv(
    "data/Resume/Resume.csv"
)

X = df["Resume_str"]
y = df["Category"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

# Model
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)

predictions = model.predict(
    X_test_tfidf
)

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(12, 10))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Resume Classification Confusion Matrix"
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png"
)

print(
    "Confusion matrix saved to results/confusion_matrix.png"
)