import streamlit as st
import pandas as pd
import joblib
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🎯",
    layout="wide"
)

# Load model and vectorizer
model = joblib.load("results/resume_classifier.pkl")
vectorizer = joblib.load("results/tfidf_vectorizer.pkl")

# Load datasets
resume_df = pd.read_csv("data/Resume/Resume.csv")
job_df = pd.read_csv("data/job_dataset.csv")

# Header
st.title("🎯 AI Career Advisor")

st.markdown("""
AI-powered resume analysis system that classifies resumes,
recommends suitable jobs, and identifies relevant skills.
""")

st.info(
    "Upload your own resume or select a sample resume from the dataset."
)

st.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Career Advisor", "About Project"]
)
if page == "About Project":

    st.title("About Project")

    st.subheader("Authors")
    st.write("Sara Hodzic")
    st.write("Mithat Misirlic")

    st.subheader("Technologies")
    st.write("""
    • Python
    • Pandas
    • Scikit-Learn
    • TF-IDF
    • Streamlit
    """)

    st.subheader("Dataset")
    st.write("""
    Resume Dataset and Job Dataset
    """)

    st.subheader("Ethical Considerations")
    st.write("""
    This system provides career recommendations
    based on resume content and should not replace
    human judgment or professional career advice.
    """)

    st.stop()

# Upload Resume
uploaded_file = st.file_uploader(
    "📄 Upload Resume (PDF or TXT)",
    type=["pdf", "txt"]
)

# Job Description
job_description = st.text_area(
    "📋 Paste Job Description",
    height=200,
    help="Paste a job advertisement to compare it with the resume"
)

# Sidebar
st.sidebar.header("Resume Input")

if uploaded_file is None:

    resume_index = st.sidebar.selectbox(
        "Dataset Resume",
        options=range(len(resume_df)),
        index=217
    )

analyze = st.sidebar.button("Analyze Resume")

if analyze:

    # Load resume
    if uploaded_file is not None:

        if uploaded_file.type == "application/pdf":

            pdf = PdfReader(uploaded_file)

            resume_text = ""

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    resume_text += page_text + "\n"

        else:

            resume_text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        actual_category = "User Uploaded Resume"

    else:

        resume_text = resume_df.iloc[
            resume_index
        ]["Resume_str"]

        actual_category = resume_df.iloc[
            resume_index
        ]["Category"]

    # Classification
    resume_vector = vectorizer.transform(
        [resume_text]
    )

    predicted_category = model.predict(
        resume_vector
    )[0]

    career_paths = {

    "INFORMATION-TECHNOLOGY": [
        "Junior Developer",
        "Software Engineer",
        "Senior Software Engineer",
        "Tech Lead",
        "Engineering Manager"
    ],

    "FINANCE": [
        "Financial Analyst",
        "Senior Analyst",
        "Finance Manager",
        "Finance Director",
        "CFO"
    ],

    "DESIGNER": [
        "Junior Designer",
        "UI/UX Designer",
        "Senior Designer",
        "Product Designer",
        "Design Manager"
    ],

    "BUSINESS-DEVELOPMENT": [
        "Business Analyst",
        "Business Development Manager",
        "Senior Manager",
        "Director",
        "VP Business Development"
    ],

    "HEALTHCARE": [
        "Healthcare Assistant",
        "Healthcare Specialist",
        "Senior Specialist",
        "Department Manager",
        "Healthcare Director"
    ]
}  

    confidence = max(
        model.predict_proba(
            resume_vector
        )[0]
    ) * 100

    resume_lower = resume_text.lower()

    # Resume ↔ Job Description Match

    job_match_score = None

    matching_keywords = []

    missing_keywords = []

    job_match_score = None

    matching_keywords = []

    missing_keywords = []

    if job_description:

        texts = [
            resume_text,
            job_description
        ]

        tfidf = vectorizer.transform(
            texts
        )

        similarity = cosine_similarity(
            tfidf[0:1],
            tfidf[1:2]
        )[0][0]

        job_match_score = similarity * 100
    # Resume Strength Score
    strength_score = 0

    if len(resume_text.split()) > 500:
        strength_score += 20

    if "education" in resume_lower:
        strength_score += 20

    if "experience" in resume_lower:
        strength_score += 20

    if "skills" in resume_lower:
        strength_score += 20

    if "project" in resume_lower:
        strength_score += 20
    results = []

    if job_match_score is not None:

        st.subheader("📌 Job Description Match"
        )

        st.progress(job_match_score / 100
        )

        st.write(
            f"Similarity Score: {job_match_score:.2f}%"
        )


    # Job Matching
    for _, job in job_df.iterrows():

        job_skills = [
            skill.strip()
            for skill in str(job["Skills"]).split(";")
        ]

        matching_skills = []

        for skill in job_skills:

            if skill.lower() in resume_lower:

                matching_skills.append(
                    skill
                )

        missing_skills = [

            skill

            for skill in job_skills

            if skill.lower()
            not in resume_lower

        ]

        score = (
            len(matching_skills)
            / len(job_skills)
        ) * 100

        results.append({

            "title": job["Title"],

            "score": score,

            "matching_skills":
                matching_skills,

            "missing_skills":
                missing_skills

        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    best_match = results[0]
    

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Actual Category",
            actual_category
        )

    with col2:
        st.metric(
            "Predicted Category",
            predicted_category
        )

    with col3:
        st.metric(
            "Best Match Score",
            f"{best_match['score']:.2f}%"
        )

    with col4: 
        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    with col5:

        if job_match_score is not None:

            st.metric(
                "JD Match",
                f"{job_match_score:.2f}%"
            )

    st.subheader("📊 Resume Strength")

    st.progress(strength_score / 100)

    st.caption(
        f"Resume Strength Score: {strength_score}/100"
    )

    # Recommendation Quality
    if best_match["score"] >= 50:
        st.success(
            "Strong recommendation generated."
        )
    elif best_match["score"] >= 30:
        st.warning(
            "Moderate recommendation generated."
        )
    else:
        st.error(
            "No strong matches found in dataset."
        )

    st.divider()

    # Chart
    st.subheader(
        "🏆 Top 5 Job Recommendations"
    )

    chart_data = pd.DataFrame({
        "Job": [r["title"] for r in results[:5]],
        "Score": [r["score"] for r in results[:5]]
    })

    st.bar_chart(
        chart_data.set_index("Job")
    )

    # Recommendations
    for i, result in enumerate(
        results[:5],
        start=1
    ):

        with st.expander(
            f"{i}. {result['title']} ({result['score']:.2f}%)"
        ):

            st.write(
                f"**Match Score:** {result['score']:.2f}%"
            )

            st.write(
                "### ✅ Matching Skills"
            )

            if result["matching_skills"]:

                for skill in result[
                    "matching_skills"
                ]:

                    st.write(
                        f"- {skill}"
                    )

            else:

                st.write(
                    "No matching skills found."
                )

            st.write(
                "### 📈 Missing Skills"
            )

            if result["missing_skills"]:

                for skill in result[
                    "missing_skills"
                ][:5]:

                    st.write(
                        f"- {skill}"
                    )

            else:

                st.write(
                    "No missing skills identified."
                )

    st.divider()

    # Suggested Skills
    st.subheader(
        "🎓 Suggested Skills To Learn"
    )

    if best_match["missing_skills"]:

        for skill in best_match[
            "missing_skills"
        ][:5]:

            st.write(
                f"• {skill}"
            )

    else:

        st.success(
            "No major skill gaps identified."
        )

    st.divider()

    st.subheader(
        "🚀 Career Path Suggestions"
    )

    if predicted_category in career_paths:

        for step in career_paths[
            predicted_category
        ]:

            st.write(
                f"➡️ {step}"
            )

    else:

        st.write(
            "Career path information not available for this category."
        )

    # Download Report
    report = f"""
AI CAREER ADVISOR REPORT

Predicted Category:
{predicted_category}

Confidence:
{confidence:.2f}%

Best Match:
{best_match['title']}

Match Score:
{best_match['score']:.2f}%

Suggested Skills:
{', '.join(best_match['missing_skills'][:5])}
"""

    st.download_button(
        "📥 Download Career Report",
        report,
        file_name="career_report.txt"
    )

    st.divider()

    # Resume Preview
    st.subheader(
        "📄 Resume Preview"
    )

    st.text_area(
        "Resume Content",
        resume_text[:1500],
        height=250
    )