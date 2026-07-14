from datetime import datetime


def generate_report(
    predicted_category,
    confidence,
    confidence_level,
    confidence_message,
    top_predictions,
    important_terms,
    best_match,
    strength_score,
    resume_level,
    job_match_score=None,
):
    """
    Build the structured career-analysis report consumed by the PDF renderer.
    """

    return {
        "generated_at": datetime.now(),
        "analysis": {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "confidence_message": confidence_message,
            "strength_score": strength_score,
            "resume_level": resume_level,
            "job_match_score": job_match_score,
        },
        "top_predictions": top_predictions,
        "important_terms": important_terms,
        "best_match": {
            "title": best_match["title"],
            "score": best_match["score"],
            "recommendation_level": best_match["recommendation_level"],
            "matching_skills": best_match.get(
                "matching_skills",
                [],
            ),
            "missing_skills": best_match.get(
                "missing_skills",
                [],
            ),
        },
    }
