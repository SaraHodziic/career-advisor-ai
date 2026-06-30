import streamlit as st
import pandas as pd
import joblib

from src.resume_loader import load_resume
from src.resume_strength import calculate_resume_strength
from src.recommender import recommend_jobs
from src.jd_matcher import calculate_job_match
from src.career_paths import career_paths
from src.report_generator import generate_report


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="💼",
    layout="wide"
)

with open("styles.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

model = joblib.load(
    "results/resume_classifier.pkl"
)

vectorizer = joblib.load(
    "results/tfidf_vectorizer.pkl"
)


# --------------------------------------------------
# Load Datasets
# --------------------------------------------------

resume_df = pd.read_csv(
    "data/Resume/Resume.csv"
)

job_df = pd.read_csv(
    "data/job_dataset.csv"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("AI Career Advisor")

st.write(
    """
    Machine Learning based career advisory system
    for resume classification, job recommendation,
    skill gap analysis and job description matching.
    """
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

page = st.sidebar.radio(

    "Navigation",

    [

        "Career Advisor",

        "About Project"

    ]

)


# --------------------------------------------------
# About Page
# --------------------------------------------------

if page == "About Project":

    st.title("About Project")

    st.subheader("Authors")

    st.write("Sara Hodzic")

    st.write("Mithat Misirlic")


    st.subheader("Technologies")

    st.write("""

- Python
- Pandas
- Scikit-Learn
- Streamlit
- TF-IDF
- Logistic Regression
- MLP Classifier

""")


    st.subheader("Datasets")

    st.write("""

Resume Dataset

- 2484 resumes
- 24 categories

Job Dataset

- 1068 job descriptions

""")


    st.subheader("Model Evaluation")

    st.write("""

Accuracy : 65.39%

Precision : 68.56%

Recall : 65.39%

F1 Score : 64.97%

Logistic Regression : 65.39%

MLP : 62.78%

""")


    st.image(
        "results/confusion_matrix.png"
    )

    st.stop()


# --------------------------------------------------
# Resume Input
# --------------------------------------------------

st.sidebar.header(
    "Resume Input"
)

uploaded_file = st.sidebar.file_uploader(

    "Upload Resume",

    type=[
        "pdf",
        "txt"
    ]

)

job_description = st.sidebar.text_area(

    "Paste Job Description",

    height=120

)

if uploaded_file is None:

    resume_index = st.sidebar.selectbox(

        "Dataset Resume",

        options=range(
            len(resume_df)
        ),

        index=217

    )

else:

    resume_index = None


analyze = st.sidebar.button(
    "Analyze Resume"
)
# --------------------------------------------------
# Resume Analysis
# --------------------------------------------------

if analyze:

    # Load Resume
    resume_text, actual_category = load_resume(
        uploaded_file,
        resume_df,
        resume_index
    )

    # Classification
    resume_vector = vectorizer.transform(
        [resume_text]
    )

    predicted_category = model.predict(
        resume_vector
    )[0]

    confidence = max(
        model.predict_proba(
            resume_vector
        )[0]
    ) * 100

    # Resume Strength
    strength_score, breakdown, resume_level = (
        calculate_resume_strength(
            resume_text
        )
    )

    # Job Description Match
    job_match_score = calculate_job_match(
        resume_text,
        job_description,
        vectorizer
    )

    # Job Recommendations
    results = recommend_jobs(
        resume_text,
        job_df
    )

    best_match = results[0]

    # Recommendation Level
    if best_match["score"] >= 70:

        recommendation_level = "Excellent"

    elif best_match["score"] >= 50:

        recommendation_level = "Good"

    elif best_match["score"] >= 30:

        recommendation_level = "Moderate"

    else:

        recommendation_level = "Weak"

    # Short Category Names
    category_names = {

        "INFORMATION-TECHNOLOGY": "IT",

        "BUSINESS-DEVELOPMENT": "Business Development",

        "PUBLIC-RELATIONS": "Public Relations"

    }

    display_actual = category_names.get(
        actual_category,
        actual_category
    )

    display_predicted = category_names.get(
        predicted_category,
        predicted_category
    )

    # ----------------------------
    # Dashboard Metrics
    # ----------------------------

    st.subheader("Analysis Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Category",
            display_predicted
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with col3:

        st.metric(
            "Resume Quality",
            resume_level
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "Best Match",
            f"{best_match['score']:.2f}%"
        )

    with col5:

        if job_match_score is not None:

            st.metric(
                "JD Match",
                f"{job_match_score:.2f}%"
            )

        else:

            st.metric(
                "JD Match",
                "N/A"
            )

    with col6:

        st.metric(
            "Recommendation",
            recommendation_level
        )

    st.divider()

    # ----------------------------
    # Resume Strength
    # ----------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Resume Strength")

        st.progress(
            strength_score / 100
        )

        st.caption(
            f"Score: {strength_score}/100"
        )

    with right:

        if job_match_score is not None:

            st.subheader(
                "Job Description Match"
            )

            st.progress(
                job_match_score / 100
            )

            st.caption(
                f"Similarity: {job_match_score:.2f}%"
            )
                # --------------------------------------------------
    # Resume Quality Breakdown
    # --------------------------------------------------

    st.divider()

    st.subheader("Resume Quality Breakdown")

    breakdown_df = pd.DataFrame(
        breakdown,
        columns=[
            "Criterion",
            "Points"
        ]
    )

    st.dataframe(
        breakdown_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # Recommendation Status
    # --------------------------------------------------

    if best_match["score"] >= 70:

        st.success(
            "Excellent job recommendation generated."
        )

    elif best_match["score"] >= 50:

        st.success(
            "Strong recommendation generated."
        )

    elif best_match["score"] >= 30:

        st.warning(
            "Moderate recommendation generated."
        )

    else:

        st.error(
            "No strong matches found."
        )

    st.divider()

    # --------------------------------------------------
    # Top 5 Job Recommendations
    # --------------------------------------------------

    st.subheader(
        "Top 5 Job Recommendations"
    )

    chart_data = pd.DataFrame({

        "Job":

        [
            r["title"]
            for r in results[:5]
        ],

        "Score":

        [
            r["score"]
            for r in results[:5]
        ]

    })

    st.bar_chart(
        chart_data.set_index("Job")
    )

    # --------------------------------------------------
    # Recommendation Details
    # --------------------------------------------------

    for i, result in enumerate(
        results[:5],
        start=1
    ):

        with st.expander(

            f"{i}. {result['title']} ({result['score']:.2f}%)"

        ):

            st.write(
                f"Match Score: {result['score']:.2f}%"
            )

            st.write("Matching Skills")

            if result["matching_skills"]:

                for skill in result[
                    "matching_skills"
                ]:

                    st.write(
                        f"- {skill}"
                    )

            else:

                st.write(
                    "No matching skills."
                )

            st.write("Missing Skills")

            if result["missing_skills"]:

                for skill in result[
                    "missing_skills"
                ][:10]:

                    st.write(
                        f"- {skill}"
                    )

            else:

                st.write(
                    "No missing skills."
                )

    st.divider()

    # --------------------------------------------------
    # Suggested Skills
    # --------------------------------------------------

    st.subheader(
        "Suggested Skills"
    )

    if best_match["missing_skills"]:

        for skill in best_match[
            "missing_skills"
        ][:10]:

            st.write(
                f"• {skill}"
            )

    else:

        st.success(
            "No significant skill gaps detected."
        )

    st.divider()

    # --------------------------------------------------
    # Career Path
    # --------------------------------------------------

    st.subheader(
        "Career Path"
    )

    if predicted_category in career_paths:

        for step in career_paths[
            predicted_category
        ]:

            st.write(
                f"→ {step}"
            )

    else:

        st.info(
            "Career path is not available for this category."
        )

    st.divider()

    # --------------------------------------------------
    # Download Report
    # --------------------------------------------------

    report = generate_report(

        predicted_category,

        confidence,

        {
            **best_match,
            "recommendation_level": recommendation_level
        },

        strength_score,

        resume_level,

        job_match_score

    )

    st.download_button(

        label="Download Career Report",

        data=report,

        file_name="career_report.txt",

        mime="text/plain"

    )

    st.divider()

    # --------------------------------------------------
    # Resume Preview
    # --------------------------------------------------

    st.subheader(
        "Resume Preview"
    )

    st.text_area(

        "Resume",

        resume_text[:2000],

        height=300

    )