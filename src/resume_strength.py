def calculate_resume_strength(resume_text):
    """
    Calculates the quality score of a resume based on
    several predefined criteria.

    Returns:
        strength_score (int)
        breakdown (list)
        level (str)
    """

    resume_lower = resume_text.lower()

    strength_score = 0
    breakdown = []

    # -------------------------
    # Resume Length
    # -------------------------
    word_count = len(resume_text.split())

    if word_count >= 700:
        strength_score += 15
        breakdown.append(("Resume Length", 15))

    elif word_count >= 400:
        strength_score += 10
        breakdown.append(("Resume Length", 10))

    elif word_count >= 200:
        strength_score += 5
        breakdown.append(("Resume Length", 5))

    else:
        breakdown.append(("Resume Length", 0))

    # -------------------------
    # Resume Sections
    # -------------------------
    sections = [
        "education",
        "experience",
        "skills",
        "projects",
        "summary",
        "certification"
    ]

    for section in sections:

        if section in resume_lower:

            strength_score += 10
            breakdown.append(
                (section.capitalize(), 10)
            )

        else:

            breakdown.append(
                (section.capitalize(), 0)
            )

    # -------------------------
    # Contact Information
    # -------------------------

    if "@" in resume_text:

        strength_score += 10
        breakdown.append(("Email", 10))

    else:

        breakdown.append(("Email", 0))

    # -------------------------
    # Phone Number
    # -------------------------

    digit_count = sum(
        c.isdigit()
        for c in resume_text
    )

    if digit_count >= 8:

        strength_score += 5
        breakdown.append(("Phone", 5))

    else:

        breakdown.append(("Phone", 0))

    # -------------------------
    # LinkedIn
    # -------------------------

    if "linkedin" in resume_lower:

        strength_score += 5
        breakdown.append(("LinkedIn", 5))

    else:

        breakdown.append(("LinkedIn", 0))

    # -------------------------
    # GitHub
    # -------------------------

    if "github" in resume_lower:

        strength_score += 5
        breakdown.append(("GitHub", 5))

    else:

        breakdown.append(("GitHub", 0))

    # -------------------------
    # Maximum Score
    # -------------------------

    strength_score = min(
        strength_score,
        100
    )

    # -------------------------
    # Resume Level
    # -------------------------

    if strength_score >= 80:

        level = "Excellent"

    elif strength_score >= 60:

        level = "Good"

    elif strength_score >= 40:

        level = "Moderate"

    else:

        level = "Weak"

    return (
        strength_score,
        breakdown,
        level
    )