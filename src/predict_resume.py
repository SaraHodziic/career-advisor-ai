import pandas as pd
import joblib

model = joblib.load("results/resume_classifier.pkl")
vectorizer = joblib.load("results/tfidf_vectorizer.pkl")

df = pd.read_csv("data/Resume/Resume.csv")

resume_index = 217

resume_text = df.iloc[resume_index]["Resume_str"]

actual_category = df.iloc[resume_index]["Category"]

resume_vector = vectorizer.transform([resume_text])

prediction = model.predict(resume_vector)[0]

print("Actual Category:")
print(actual_category)

print("\nPredicted Category:")
print(prediction)