# AI Career Advisor

## Overview

AI Career Advisor is a machine learning-based web application designed to assist users in career planning through resume analysis.

The system can:

- Classify resumes into professional categories
- Recommend suitable job positions
- Identify missing skills
- Match resumes against job descriptions
- Suggest potential career paths

---

## Datasets

### Resume Dataset
- 2,484 resumes
- 24 professional categories

### Job Dataset
- 1,068 job descriptions
- Skills extracted for recommendation and matching

---

## Technologies

- Python
- Pandas
- Scikit-Learn
- Streamlit
- TF-IDF Vectorization
- Logistic Regression
- Multi-Layer Perceptron (MLP)

---

## Machine Learning Pipeline

1. Resume preprocessing
2. TF-IDF feature extraction
3. Resume classification using Logistic Regression
4. Job recommendation based on skill matching
5. Resume–Job Description similarity using cosine similarity

---

## Features

- Resume Classification
- Job Recommendation Engine
- Skill Gap Analysis
- Resume Strength Assessment
- Job Description Matching
- Career Path Suggestions
- Downloadable Career Report

---

## Evaluation Results

### Logistic Regression

| Metric | Score |
|----------|----------|
| Accuracy | 65.39% |
| Precision | 68.56% |
| Recall | 65.39% |
| F1 Score | 64.97% |

### Model Comparison

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 65.39% |
| MLP Classifier | 62.78% |

The Logistic Regression model achieved the best overall performance and was selected as the final model.

---

## Project Structure

text career-advisor-ai/ │ ├── app.py ├── data/ ├── results/ ├── src/ ├── styles.css └── requirements.txt 

---

## How to Run

Install dependencies:

bash pip install -r requirements.txt 

Run the application:

bash streamlit run app.py 

---

## Authors

- Sara Hodzic
- Mithat Misirlic