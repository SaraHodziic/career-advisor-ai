import pandas as pd

resume_df = pd.read_csv("data/Resume/Resume.csv")
job_df = pd.read_csv("data/job_dataset.csv")


def match_resume_to_job(resume_index, job_index):

    resume_text = resume_df.iloc[resume_index]["Resume_str"].lower()

    job = job_df.iloc[job_index]

    job_skills = [
        skill.strip()
        for skill in str(job["Skills"]).split(";")
    ]

    matching_skills = []
    missing_skills = []

    for skill in job_skills:
        if skill.lower() in resume_text:
            matching_skills.append(skill)
        else:
            missing_skills.append(skill)

    match_percentage = (
        len(matching_skills) / len(job_skills)
    ) * 100

    print("\nJOB:")
    print(job["Title"])

    print("\nMATCHING SKILLS:")
    for skill in matching_skills:
        print("-", skill)

    print("\nMISSING SKILLS:")
    for skill in missing_skills:
        print("-", skill)

    print(f"\nSKILL MATCH: {match_percentage:.2f}%")

    print("\nRECOMMENDATION:")

    if match_percentage >= 70:
        print("Strong match for this position.")
    elif match_percentage >= 40:
        print("Moderate match. Additional skills may be required.")
    else:
        print("Not a strong match for this position.")
        print("\nSuggested skills to learn:")

        for skill in missing_skills[:5]:
            print("-", skill)


match_resume_to_job(217, 0)