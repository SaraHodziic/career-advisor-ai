import pandas as pd


def recommend_jobs(resume_text, job_df):

    resume_lower = resume_text.lower()

    results = []

    for _, job in job_df.iterrows():

        job_skills = [
            skill.strip()
            for skill in str(job["Skills"]).split(";")
        ]

        matching_skills = []

        for skill in job_skills:

            if skill.lower() in resume_lower:

                matching_skills.append(skill)

        missing_skills = [

            skill

            for skill in job_skills

            if skill.lower() not in resume_lower

        ]

        score = (
            len(matching_skills)
            / len(job_skills)
        ) * 100

        results.append({

            "title": job["Title"],

            "score": score,

            "matching_skills": matching_skills,

            "missing_skills": missing_skills

        })

    results = sorted(

        results,

        key=lambda x: x["score"],

        reverse=True

    )

    return results