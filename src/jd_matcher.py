from sklearn.metrics.pairwise import cosine_similarity


def calculate_job_match(
    resume_text,
    job_description,
    vectorizer,
):
    """
    Calculate resume-to-job-description similarity.

    Returns:
        float | None: Similarity score from 0 to 100, or None when no
        job description is provided.
    """

    if not job_description:
        return None

    texts = [
        resume_text,
        job_description,
    ]

    tfidf = vectorizer.transform(texts)
    similarity = cosine_similarity(
        tfidf[0:1],
        tfidf[1:2],
    )[0][0]

    return similarity * 100
