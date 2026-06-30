import streamlit as st
import pandas as pd
import joblib

from src.resume_loader import load_resume
from src.resume_strength import calculate_resume_strength
from src.recommender import recommend_jobs
from src.jd_matcher import calculate_job_match
from src.career_paths import career_paths
from src.report_generator import generate_report
from src.pdf_report import create_pdf

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

    st.subheader("Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Resumes", "2,484")

    with col2:
        st.metric("Job Descriptions", "1,068")

    with col3:
        st.metric("Job Categories", "24")

    with col4:
        st.metric("ML Models", "2")

    st.subheader("Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", "65.39%")
        st.metric("Precision", "68.56%")

    with col2:
        st.metric("Recall", "65.39%")
        st.metric("F1 Score", "64.97%")

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "MLP Classifier"
        ],
        "Accuracy": [
            65.39,
            62.78
        ]
    })

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Confusion Matrix")

    st.image("results/confusion_matrix.png")

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
    with st.spinner("Analyzing Resume..."):
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

        # --------------------------------------------------
    # Resume Quality Breakdown
    # --------------------------------------------------

    st.subheader("Resume Quality Breakdown")

    breakdown_df = pd.DataFrame(
    breakdown,
    columns=["Criterion", "Points"]
)

    breakdown_df["Status"] = breakdown_df["Points"].apply(
    lambda x: "Passed" if x > 0 else "Missing"
)

    breakdown_df = breakdown_df[
    ["Criterion", "Status", "Points"]
]

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
        f"Top {min(5, len(results))} Job Recommendations"
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

    st.subheader(f"Top {min(5, len(results))} Job Recommendations")

    for idx, result in enumerate(results[:5], start=1):

        with st.container(border=True):

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"### {idx}. {result['title']}")

            with col2:
                st.metric(
                    "Match",
                    f"{result['score']:.1f}%"
                )

            st.progress(result["score"] / 100)

            st.markdown("#### Matching Skills")

            if result["matching_skills"]:

                cols = st.columns(3)

                for j, skill in enumerate(result["matching_skills"][:9]):
                    cols[j % 3].success(skill)

            else:
                st.write("No matching skills found.")

            st.markdown("#### Missing Skills")

            if result["missing_skills"]:

                cols = st.columns(3)

                for j, skill in enumerate(result["missing_skills"][:9]):
                    cols[j % 3].warning(skill)

            else:
                st.write("No missing skills.")

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
    pdf = create_pdf(report)

    st.download_button(
    "Download Career Report (PDF)",
    data=pdf,
    file_name="career_report.pdf",
    mime="application/pdf"
)

    st.divider()
# --------------------------------------------------
# Resume Preview
# --------------------------------------------------

    with st.expander("Resume Preview"):

        preview = resume_text[:2000]

        st.text_area(
          "",
           preview,
           height=350,
         disabled=True,
         label_visibility="collapsed"
    )

        if len(resume_text) > 2000:
         st.caption("Showing the first 2000 characters.")

st.divider()
st.caption(
    "AI Career Advisor | SRH University of Applied Sciences | 2026"
)