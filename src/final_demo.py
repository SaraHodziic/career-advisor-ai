import pandas as pd
import joblib

# Load saved model and vectorizer
model = joblib.load("results/resume_classifier.pkl")
vectorizer = joblib.load("results/tfidf_vectorizer.pkl")

# Load datasets
resume_df = pd.read_csv("data/Resume/Resume.csv")
job_df = pd.read_csv("data/job_dataset.csv")

# Choose a resume
resume_index = 217

resume_text = resume_df.iloc[resume_index]["Resume_str"]
actual_category = resume_df.iloc[resume_index]["Category"]

# Predict category
resume_vector = vectorizer.transform([resume_text])
predicted_category = model.predict(resume_vector)[0]

print("=" * 50)
print("AI CAREER ADVISOR")
print("=" * 50)

print(f"\nActual Category: {actual_category}")
print(f"Predicted Category: {predicted_category}")

# Recommend jobs
results = []

resume_lower = resume_text.lower()

for _, job in job_df.iterrows():

    job_skills = [
        skill.strip()
        for skill in str(job["Skills"]).split(";")
    ]

    matching_skills = []

    for skill in job_skills:
        if skill.lower() in resume_lower:
            matching_skills.append(skill)

    score = (len(matching_skills) / len(job_skills)) * 100

    results.append({
        "title": job["Title"],
        "score": score,
        "skills": matching_skills
    })

results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

print("\nTOP 5 JOB RECOMMENDATIONS:\n")

for result in results[:5]:

    print(f"{result['title']} - {result['score']:.2f}%")

    if result["skills"]:
        print("Matching Skills:")

        for skill in result["skills"]:
            print(f"  - {skill}")

    print()
    