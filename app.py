import joblib
import pandas as pd
import streamlit as st

from src.career_paths import career_paths
from src.jd_matcher import calculate_job_match
from src.pdf_report import create_pdf
from src.recommender import recommend_jobs
from src.report_generator import generate_report
from src.resume_loader import load_resume
from src.resume_strength import calculate_resume_strength


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("styles.css", encoding="utf-8") as stylesheet:
    st.markdown(
        f"<style>{stylesheet.read()}</style>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Resource Loading
# --------------------------------------------------

@st.cache_resource
def load_models():
    classifier = joblib.load("results/resume_classifier.pkl")
    tfidf_vectorizer = joblib.load("results/tfidf_vectorizer.pkl")
    return classifier, tfidf_vectorizer


@st.cache_data
def load_datasets():
    resumes = pd.read_csv("data/Resume/Resume.csv")
    jobs = pd.read_csv("data/job_dataset.csv")
    return resumes, jobs


model, vectorizer = load_models()
resume_df, job_df = load_datasets()


# --------------------------------------------------
# Prediction Explanation (XAI)
# --------------------------------------------------

def get_prediction_explanation(
    classifier,
    tfidf_vectorizer,
    resume_vector,
    predicted_category,
    top_n=8,
):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    class_index = list(classifier.classes_).index(predicted_category)
    class_coefficients = classifier.coef_[class_index]
    resume_values = resume_vector.toarray()[0]
    contributions = resume_values * class_coefficients
    top_indices = contributions.argsort()[::-1]

    important_terms = []

    for index in top_indices:
        if contributions[index] <= 0:
            continue

        important_terms.append(
            {
                "term": feature_names[index],
                "contribution": float(contributions[index]),
            }
        )

        if len(important_terms) == top_n:
            break

    return important_terms


# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------

st.sidebar.markdown("## Resume Analysis")
st.sidebar.caption(
    "Upload your own resume or select an example from the project dataset."
)

with st.sidebar.container(border=True):
    st.markdown("#### 1. Select Resume")

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        help=(
            "Upload a text-based PDF or TXT resume. "
            "Scanned PDFs may not contain readable text."
        ),
    )

    if uploaded_file is None:
        resume_index = st.selectbox(
            "Or choose a dataset resume",
            options=range(len(resume_df)),
            index=min(217, len(resume_df) - 1),
            format_func=lambda index: (
                f"Resume {index + 1} — {resume_df.iloc[index]['Category']}"
            ),
        )
    else:
        resume_index = None

with st.sidebar.container(border=True):
    st.markdown("#### 2. Optional Job Description")

    job_description = st.text_area(
        "Paste a job description",
        height=150,
        placeholder=(
            "Paste a job description to calculate "
            "resume-to-job similarity..."
        ),
        help=(
            "This is optional. The application uses TF-IDF "
            "and cosine similarity for matching."
        ),
    )

analyze = st.sidebar.button(
    "Analyse Resume",
    type="primary",
    use_container_width=True,
)

# Navigation is deliberately rendered last so it appears at the bottom.
st.sidebar.markdown('<div class="sidebar-navigation-spacer"></div>', unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption("Navigation")

page = st.sidebar.radio(
    "Navigation",
    ["Career Advisor", "About Project"],
    label_visibility="collapsed",
)


# --------------------------------------------------
# Hero Section
# --------------------------------------------------

st.html(
    """
<div class="hero-container">
    <div class="hero-content">
        <div class="hero-badge">AI-powered resume intelligence</div>

        <h1 class="hero-title">
            Turn your resume into
            <span class="hero-gradient-text">career direction.</span>
        </h1>

        <p class="hero-description">
            Classify your professional profile, discover matching roles,
            uncover missing skills and understand why the model made its prediction.
        </p>

        <div class="hero-meta">
            <span>24 career categories</span>
            <span>2,484 training resumes</span>
            <span>Explainable AI insights</span>
        </div>
    </div>

    <div class="hero-orb hero-orb-one"></div>
    <div class="hero-orb hero-orb-two"></div>
</div>
"""
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
    st.markdown(
        """
- Python
- Pandas
- Scikit-Learn
- Streamlit
- TF-IDF
- Logistic Regression
- MLP Classifier
"""
    )

    st.subheader("Project Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Resumes", f"{len(resume_df):,}")

    with stat_col2:
        st.metric("Job Descriptions", f"{len(job_df):,}")

    with stat_col3:
        st.metric("Job Categories", resume_df["Category"].nunique())

    with stat_col4:
        st.metric("ML Models", "2")

    st.subheader("Model Evaluation")
    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.metric("Accuracy", "65.39%")
        st.metric("Precision", "68.56%")

    with eval_col2:
        st.metric("Recall", "65.39%")
        st.metric("F1 Score", "64.97%")

    st.subheader("Model Comparison")

    comparison = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "MLP Classifier"],
            "Accuracy": [65.39, 62.78],
        }
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Confusion Matrix")
    st.image("results/confusion_matrix.png")

    st.stop()


# --------------------------------------------------
# Career Advisor Page
# --------------------------------------------------

if analyze:
    with st.spinner("Analyzing Resume..."):
        resume_text, actual_category = load_resume(
            uploaded_file,
            resume_df,
            resume_index,
        )

    if not resume_text or not resume_text.strip():
        st.error("No readable resume text was found.")
        st.stop()

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    resume_vector = vectorizer.transform([resume_text])
    probabilities = model.predict_proba(resume_vector)[0]
    classes = model.classes_
    top_indices = probabilities.argsort()[-3:][::-1]

    top_predictions = [
        {
            "category": classes[index],
            "confidence": probabilities[index] * 100,
        }
        for index in top_indices
    ]

    predicted_category = top_predictions[0]["category"]
    confidence = top_predictions[0]["confidence"]

    # --------------------------------------------------
    # Confidence Interpretation
    # --------------------------------------------------

    if confidence >= 70:
        confidence_level = "High Confidence"
        confidence_message = (
            "The model found strong evidence supporting "
            "the predicted professional category."
        )

    elif confidence >= 40:
        confidence_level = "Moderate Confidence"
        confidence_message = (
            "The resume contains relevant evidence, but it also "
            "shares characteristics with other categories."
        )

    else:
        confidence_level = "Low Confidence"
        confidence_message = (
            "The resume spans several professional areas, so the "
            "model cannot assign one category with strong certainty."
        )

# --------------------------------------------------
# Confidence Notice
# --------------------------------------------------

        if confidence_level == "High Confidence":
            st.success(
                f"**{confidence_level} — {confidence:.2f}%**\n\n"
                f"{confidence_message}"
            )

        elif confidence_level == "Moderate Confidence":
            st.warning(
                f"**{confidence_level} — {confidence:.2f}%**\n\n"
                f"{confidence_message}"
            )

        else:
            st.error(
                f"**{confidence_level} — {confidence:.2f}%**\n\n"
                f"{confidence_message}"
            )

    important_terms = get_prediction_explanation(
        classifier=model,
        tfidf_vectorizer=vectorizer,
        resume_vector=resume_vector,
        predicted_category=predicted_category,
        top_n=8,
    )

    # --------------------------------------------------
    # Supporting Analysis
    # --------------------------------------------------

    strength_score, breakdown, resume_level = calculate_resume_strength(
        resume_text
    )

    job_match_score = calculate_job_match(
        resume_text,
        job_description,
        vectorizer,
    )

    results = recommend_jobs(resume_text, job_df)

    if not results:
        st.warning("No job recommendations could be generated for this resume.")
        st.stop()

    best_match = results[0]

    if best_match["score"] >= 70:
        recommendation_level = "Excellent"
    elif best_match["score"] >= 50:
        recommendation_level = "Good"
    elif best_match["score"] >= 30:
        recommendation_level = "Moderate"
    else:
        recommendation_level = "Weak"

    category_names = {
        "INFORMATION-TECHNOLOGY": "IT",
        "BUSINESS-DEVELOPMENT": "Business Development",
        "PUBLIC-RELATIONS": "Public Relations",
    }

    display_predicted = category_names.get(
        predicted_category,
        predicted_category,
    )

    # --------------------------------------------------
    # Analysis Summary
    # --------------------------------------------------

    st.subheader("Analysis Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("Primary Category", display_predicted)

    with summary_col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    with summary_col3:
        st.metric("Resume Quality", resume_level)

    st.subheader("Top 3 Predicted Categories")
    prediction_columns = st.columns(3)

    for column, prediction in zip(prediction_columns, top_predictions):
        display_name = category_names.get(
            prediction["category"],
            prediction["category"],
        )

        with column:
            st.metric(
                display_name,
                f"{prediction['confidence']:.2f}%",
            )

    # --------------------------------------------------
    # Prediction Explanation
    # --------------------------------------------------

    st.subheader("Why This Prediction?")

    if important_terms:
        explanation_df = pd.DataFrame(important_terms)
        explanation_df.columns = ["Feature", "Contribution"]

        st.bar_chart(explanation_df.set_index("Feature"))
        st.dataframe(
            explanation_df,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "These resume terms contributed the most to the predicted category."
        )
    else:
        st.info("No important prediction terms were found.")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Best Match", f"{best_match['score']:.2f}%")

    with result_col2:
        st.metric(
            "JD Match",
            f"{job_match_score:.2f}%" if job_match_score is not None else "N/A",
        )

    with result_col3:
        st.metric("Recommendation", recommendation_level)

    st.divider()

    # --------------------------------------------------
    # Resume Quality
    # --------------------------------------------------

    strength_col, jd_col = st.columns(2)

    with strength_col:
        st.subheader("Resume Strength")
        st.progress(strength_score / 100)
        st.caption(f"Score: {strength_score}/100")

    with jd_col:
        if job_match_score is not None:
            st.subheader("Job Description Match")
            st.progress(job_match_score / 100)
            st.caption(f"Similarity: {job_match_score:.2f}%")

    st.subheader("Resume Quality Breakdown")

    breakdown_df = pd.DataFrame(
        breakdown,
        columns=["Criterion", "Points"],
    )

    breakdown_df["Status"] = breakdown_df["Points"].apply(
        lambda points: "Passed" if points > 0 else "Missing"
    )

    breakdown_df = breakdown_df[["Criterion", "Status", "Points"]]

    st.dataframe(
        breakdown_df,
        use_container_width=True,
        hide_index=True,
    )

    if best_match["score"] >= 70:
        st.success("Excellent job recommendation generated.")
    elif best_match["score"] >= 50:
        st.success("Strong recommendation generated.")
    elif best_match["score"] >= 30:
        st.warning("Moderate recommendation generated.")
    else:
        st.error("No strong matches found.")

    st.divider()

    # --------------------------------------------------
    # Job Recommendations
    # --------------------------------------------------

    recommendation_count = min(5, len(results))
    st.subheader(f"Top {recommendation_count} Job Recommendations")

    chart_data = pd.DataFrame(
        {
            "Job": [result["title"] for result in results[:5]],
            "Score": [result["score"] for result in results[:5]],
        }
    )

    st.bar_chart(chart_data.set_index("Job"))

    for index, result in enumerate(results[:5], start=1):
        score = result["score"]

        if score >= 70:
            score_label = "Excellent match"
        elif score >= 50:
            score_label = "Good match"
        elif score >= 30:
            score_label = "Moderate match"
        else:
            score_label = "Weak match"

        with st.expander(
            f"{index}. {result['title']} — {score:.1f}%",
            expanded=(index == 1),
        ):
            title_column, score_column = st.columns(
                [4, 1],
                vertical_alignment="center",
            )

            with title_column:
                st.markdown(f"### {result['title']}")
                st.caption(score_label)

            with score_column:
                st.metric("Match", f"{score:.1f}%")

            st.progress(min(max(score / 100, 0.0), 1.0))

            matching_column, missing_column = st.columns(2, gap="large")

            with matching_column:
                st.markdown("#### Matching skills")

                if result["matching_skills"]:
                    st.markdown(
                        "\n".join(
                            f"- ✅ {skill}"
                            for skill in result["matching_skills"][:12]
                        )
                    )
                else:
                    st.info("No explicitly matching skills found.")

            with missing_column:
                st.markdown("#### Skills to develop")

                if result["missing_skills"]:
                    st.markdown(
                        "\n".join(
                            f"- ◻️ {skill}"
                            for skill in result["missing_skills"][:12]
                        )
                    )
                else:
                    st.success("No missing skills identified.")

    # --------------------------------------------------
    # Career Path
    # --------------------------------------------------

    st.subheader("Suggested Career Path")

    if predicted_category in career_paths:
        path_steps = career_paths[predicted_category]
        path_columns = st.columns(len(path_steps))

        for index, step in enumerate(path_steps):
            with path_columns[index]:
                st.html(
                    f"""
<div class="career-step">
    <div class="career-step-number">{index + 1}</div>
    <div class="career-step-title">{step}</div>
</div>
"""
                )
    else:
        st.info(
            "A predefined career path is not yet available for this category."
        )

    st.divider()

    # --------------------------------------------------
    # Download Report
    # --------------------------------------------------

    report = generate_report(
        predicted_category=predicted_category,
        confidence=confidence,
        confidence_level=confidence_level,
        confidence_message=confidence_message,
        top_predictions=top_predictions,
        important_terms=important_terms,
        best_match={
            **best_match,
            "recommendation_level": recommendation_level,
        },
        strength_score=strength_score,
        resume_level=resume_level,
        job_match_score=job_match_score,
    )

    pdf = create_pdf(report)

    st.download_button(
        "Download Official Career Report",
        data=pdf,
        file_name="ai_career_analysis_report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
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
            label_visibility="collapsed",
        )

        if len(resume_text) > 2000:
            st.caption("Showing the first 2000 characters.")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(
    "AI Career Advisor | SRH University of Applied Sciences | 2026"
)