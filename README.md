# AI Career Advisor

AI Career Advisor is a machine learning-based web application that analyzes resumes and provides intelligent career recommendations.

The application classifies resumes into professional categories, recommends suitable job positions, identifies missing skills, evaluates resume quality, and compares resumes with job descriptions using Natural Language Processing techniques.

---

# Features

- Resume Classification
- Job Recommendation Engine
- Resume Strength Assessment
- Skill Gap Analysis
- Job Description Matching
- Career Path Suggestions
- Downloadable Career Report

---

# Datasets

## Resume Dataset

- 2,484 resumes
- 24 professional categories

## Job Dataset

- 1,068 job descriptions
- Skill-based job recommendation dataset

---

# Technologies

- Python
- Pandas
- Scikit-Learn
- Streamlit
- TF-IDF Vectorization
- Logistic Regression
- Multi-Layer Perceptron (MLP)

---

# Machine Learning Pipeline

1. Resume preprocessing
2. TF-IDF feature extraction
3. Resume classification using Logistic Regression
4. Resume strength evaluation
5. Job recommendation based on skill matching
6. Resume–Job Description similarity using Cosine Similarity
7. Career path generation

---

# Model Evaluation

## Logistic Regression

| Metric | Score |
|--------|-------:|
| Accuracy | **65.39%** |
| Precision | **68.56%** |
| Recall | **65.39%** |
| F1 Score | **64.97%** |

## Model Comparison

| Model | Accuracy |
|--------|----------:|
| Logistic Regression | **65.39%** |
| MLP Classifier | **62.78%** |

The Logistic Regression model achieved the highest overall performance and was selected as the final model for deployment.

---

# Project Structure

```text
career-advisor-ai/
│
├── app.py
├── README.md
├── requirements.txt
├── styles.css
│
├── data/
│   ├── Resume/
│   │   └── Resume.csv
│   └── job_dataset.csv
│
├── results/
│   ├── resume_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── confusion_matrix.png
│   ├── evaluation_results.txt
│   └── evaluation_table.txt
│
└── src/
    ├── resume_loader.py
    ├── resume_strength.py
    ├── recommender.py
    ├── jd_matcher.py
    ├── career_paths.py
    ├── report_generator.py
    ├── train_model.py
    ├── evaluate_recommendations.py
    ├── model_comparison.py
    ├── mlp_experiment.py
    └── confusion_matrix_plot.py
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/SaraHodziic/career-advisor-ai.git
```

Navigate to the project directory:

```bash
cd career-advisor-ai
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# Future Improvements

- PDF report generation
- Improved UI/UX
- Additional machine learning models
- More advanced resume analysis
- Enhanced recommendation engine

---

# Authors

- Sara Hodzic
- Mithat Misirlic