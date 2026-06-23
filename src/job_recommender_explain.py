import pandas as pd

resume_df = pd.read_csv("data/Resume/Resume.csv")
job_df = pd.read_csv("data/job_dataset.csv")

resume_index = 677

resume_text = resume_df.iloc[resume_index]["Resume_str"].lower()

print("Resume Category:")
print(resume_df.iloc[resume_index]["Category"])

results = []

for _, job in job_df.iterrows():

    job_skills = [
        skill.strip()
        for skill in str(job["Skills"]).split(";")
]

    matching_skills = []

    for skill in job_skills:
        if skill.lower() in resume_text:
            matching_skills.append(skill)

    match_percentage = (
        len(matching_skills) / len(job_skills)
    ) * 100

    results.append({
        "title": job["Title"],
        "score": match_percentage,
        "matching_skills": matching_skills
    })

results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

print("\nTOP 5 JOB MATCHES:\n")
best_score = results[0]["score"]

print(f"\nBEST MATCH SCORE: {best_score:.2f}%")

if best_score < 30:
    print("\nWARNING:")
    print("No strong job matches found in the dataset.")

for result in results[:5]:

    print(f"{result['title']}: {result['score']:.2f}%")

    if result["matching_skills"]:
        print("Matching Skills:")

        for skill in result["matching_skills"]:
            print("-", skill)

    print()