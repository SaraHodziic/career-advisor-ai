import pandas as pd
import joblib

# Load model
model = joblib.load(
    "results/resume_classifier.pkl"
)

vectorizer = joblib.load(
    "results/tfidf_vectorizer.pkl"
)

# Load data
resume_df = pd.read_csv(
    "data/Resume/Resume.csv"
)

job_df = pd.read_csv(
    "data/job_dataset.csv"
)

total = 0
correct = 0

# Test first 100 resumes
for _, resume in resume_df.head(100).iterrows():

    resume_text = resume["Resume_str"]

    actual_category = resume["Category"]

    resume_vector = vectorizer.transform(
        [resume_text]
    )

    predicted_category = model.predict(
        resume_vector
    )[0]

    if predicted_category == actual_category:
        correct += 1

    total += 1

accuracy = (
    correct / total
) * 100

print(
    f"Recommendation Accuracy: {accuracy:.2f}%"
)