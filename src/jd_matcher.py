from sklearn.metrics.pairwise import cosine_similarity


def calculate_job_match(
    resume_text,
    job_description,
    vectorizer
):
    """
    Calculates Resume ↔ Job Description similarity.

    Returns:
        similarity score (0-100)
    """

    if not job_description:

        return None

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

    return similarity * 100