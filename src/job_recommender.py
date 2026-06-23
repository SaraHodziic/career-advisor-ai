import pandas as pd

resume_df = pd.read_csv("data/Resume/Resume.csv")
job_df = pd.read_csv("data/job_dataset.csv")

resume_text = resume_df.iloc[217]["Resume_str"].lower()

results = []

for _, job in job_df.iterrows():

    job_skills = [
        skill.strip()
        for skill in str(job["Skills"]).split(";")
    ]

    matching_skills = 0

    for skill in job_skills:
        if skill.lower() in resume_text:
            matching_skills += 1

    match_percentage = (
        matching_skills / len(job_skills)
    ) * 100

    results.append(
        (
            job["Title"],
            match_percentage
        )
    )

results.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nTOP 5 JOB MATCHES:\n")

for title, score in results[:5]:
    print(f"{title}: {score:.2f}%")