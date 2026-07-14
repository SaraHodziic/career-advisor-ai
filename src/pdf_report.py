from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import (
    TA_CENTER,
    TA_LEFT,
    TA_RIGHT,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import cm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# --------------------------------------------------
# Document Colours
# --------------------------------------------------

NAVY = colors.HexColor("#172033")
DARK_NAVY = colors.HexColor("#111827")
INDIGO = colors.HexColor("#6366F1")
LIGHT_INDIGO = colors.HexColor("#EEF2FF")
CYAN = colors.HexColor("#22D3EE")

TEXT = colors.HexColor("#1F2937")
MUTED_TEXT = colors.HexColor("#64748B")

LIGHT_BACKGROUND = colors.HexColor("#F8FAFC")
BORDER = colors.HexColor("#E2E8F0")

SUCCESS = colors.HexColor("#15803D")
SUCCESS_BACKGROUND = colors.HexColor("#F0FDF4")

WARNING = colors.HexColor("#B45309")
WARNING_BACKGROUND = colors.HexColor("#FFFBEB")

DANGER = colors.HexColor("#B91C1C")
DANGER_BACKGROUND = colors.HexColor("#FEF2F2")


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def safe_text(value):
    """
    Convert a value into escaped text suitable for ReportLab
    Paragraph objects.
    """

    if value is None:
        return "Not provided"

    return escape(str(value))


def format_percentage(value):
    """
    Format a numeric value as a percentage.
    """

    if value is None:
        return "Not provided"

    return f"{float(value):.2f}%"


def confidence_colours(confidence_level):
    """
    Select colours based on confidence classification.
    """

    if confidence_level == "High Confidence":
        return SUCCESS_BACKGROUND, SUCCESS

    if confidence_level == "Moderate Confidence":
        return WARNING_BACKGROUND, WARNING

    return DANGER_BACKGROUND, DANGER


# --------------------------------------------------
# Header and Footer
# --------------------------------------------------

def draw_page_frame(
    canvas: Canvas,
    document: SimpleDocTemplate,
):
    """
    Draw a professional header, footer and page number.
    """

    canvas.saveState()

    page_width, page_height = A4

    # Header line
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(
        document.leftMargin,
        page_height - 1.25 * cm,
        page_width - document.rightMargin,
        page_height - 1.25 * cm,
    )

    canvas.setFillColor(NAVY)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(
        document.leftMargin,
        page_height - 0.95 * cm,
        "AI CAREER ADVISOR",
    )

    canvas.setFillColor(MUTED_TEXT)
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(
        page_width - document.rightMargin,
        page_height - 0.95 * cm,
        "Career Analysis Report",
    )

    # Footer line
    canvas.setStrokeColor(BORDER)
    canvas.line(
        document.leftMargin,
        1.15 * cm,
        page_width - document.rightMargin,
        1.15 * cm,
    )

    canvas.setFillColor(MUTED_TEXT)
    canvas.setFont("Helvetica", 8)

    canvas.drawString(
        document.leftMargin,
        0.75 * cm,
        "SRH University of Applied Sciences · 2026",
    )

    canvas.drawRightString(
        page_width - document.rightMargin,
        0.75 * cm,
        f"Page {canvas.getPageNumber()}",
    )

    canvas.restoreState()


# --------------------------------------------------
# Style Configuration
# --------------------------------------------------

def build_styles():
    """
    Create the paragraph styles used throughout the report.
    """

    styles = getSampleStyleSheet()

    return {
        "document_title": ParagraphStyle(
            "DocumentTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=32,
            textColor=DARK_NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "document_subtitle": ParagraphStyle(
            "DocumentSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=17,
            textColor=MUTED_TEXT,
            spaceAfter=22,
        ),
        "section_title": ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=10,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=TEXT,
        ),
        "body_muted": ParagraphStyle(
            "BodyMuted",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=MUTED_TEXT,
        ),
        "label": ParagraphStyle(
            "Label",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=MUTED_TEXT,
            spaceAfter=3,
        ),
        "value": ParagraphStyle(
            "Value",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=NAVY,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
        ),
        "table_text": ParagraphStyle(
            "TableText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=TEXT,
        ),
        "skill_text": ParagraphStyle(
            "SkillText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=13,
            textColor=TEXT,
        ),
        "footer_note": ParagraphStyle(
            "FooterNote",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=12,
            alignment=TA_CENTER,
            textColor=MUTED_TEXT,
        ),
    }


# --------------------------------------------------
# Reusable Components
# --------------------------------------------------

def section_heading(title, styles):
    """
    Build a consistent section heading.
    """

    return [
        Spacer(1, 0.15 * cm),
        Paragraph(
            safe_text(title),
            styles["section_title"],
        ),
        HRFlowable(
            width="100%",
            thickness=0.7,
            color=BORDER,
            spaceAfter=0.3 * cm,
        ),
    ]


def build_summary_cards(analysis, styles):
    """
    Build the principal analysis metrics.
    """

    job_match = analysis.get("job_match_score")

    card_data = [
        [
            Paragraph(
                "PREDICTED CATEGORY",
                styles["label"],
            ),
            Paragraph(
                "MODEL CONFIDENCE",
                styles["label"],
            ),
            Paragraph(
                "RESUME QUALITY",
                styles["label"],
            ),
        ],
        [
            Paragraph(
                safe_text(
                    analysis["predicted_category"]
                ),
                styles["value"],
            ),
            Paragraph(
                format_percentage(
                    analysis["confidence"]
                ),
                styles["value"],
            ),
            Paragraph(
                safe_text(
                    analysis["resume_level"]
                ),
                styles["value"],
            ),
        ],
        [
            Paragraph(
                "RESUME STRENGTH",
                styles["label"],
            ),
            Paragraph(
                "JOB DESCRIPTION MATCH",
                styles["label"],
            ),
            Paragraph(
                "REPORT STATUS",
                styles["label"],
            ),
        ],
        [
            Paragraph(
                f"{analysis['strength_score']}/100",
                styles["value"],
            ),
            Paragraph(
                format_percentage(job_match)
                if job_match is not None
                else "Not analysed",
                styles["value"],
            ),
            Paragraph(
                "Completed",
                styles["value"],
            ),
        ],
    ]

    summary_table = Table(
        card_data,
        colWidths=[
            5.45 * cm,
            5.45 * cm,
            5.45 * cm,
        ],
        rowHeights=[
            0.55 * cm,
            0.9 * cm,
            0.55 * cm,
            0.9 * cm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_BACKGROUND,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.75,
                    BORDER,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    return summary_table


def build_confidence_box(analysis, styles):
    """
    Build the confidence interpretation notice.
    """

    confidence_level = analysis["confidence_level"]

    background, foreground = confidence_colours(
        confidence_level
    )

    confidence_content = Paragraph(
        (
            f"<b>{safe_text(confidence_level)} — "
            f"{format_percentage(analysis['confidence'])}</b>"
            f"<br/><br/>"
            f"{safe_text(analysis['confidence_message'])}"
        ),
        styles["body"],
    )

    confidence_table = Table(
        [[confidence_content]],
        colWidths=[16.35 * cm],
    )

    confidence_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    background,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, -1),
                    foreground,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.75,
                    foreground,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    12,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
            ]
        )
    )

    return confidence_table


def build_prediction_table(
    predictions,
    styles,
):
    """
    Build the top-category probability table.
    """

    rows = [
        [
            Paragraph(
                "Rank",
                styles["table_header"],
            ),
            Paragraph(
                "Professional Category",
                styles["table_header"],
            ),
            Paragraph(
                "Confidence",
                styles["table_header"],
            ),
        ]
    ]

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        rows.append(
            [
                Paragraph(
                    str(index),
                    styles["table_text"],
                ),
                Paragraph(
                    safe_text(
                        prediction["category"]
                    ),
                    styles["table_text"],
                ),
                Paragraph(
                    format_percentage(
                        prediction["confidence"]
                    ),
                    styles["table_text"],
                ),
            ]
        )

    table = Table(
        rows,
        colWidths=[
            1.4 * cm,
            11.2 * cm,
            3.75 * cm,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    NAVY,
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.white,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.75,
                    BORDER,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    return table


def build_feature_table(
    important_terms,
    styles,
):
    """
    Build the explainability feature table.
    """

    rows = [
        [
            Paragraph(
                "Resume Feature",
                styles["table_header"],
            ),
            Paragraph(
                "Model Contribution",
                styles["table_header"],
            ),
        ]
    ]

    for item in important_terms:
        rows.append(
            [
                Paragraph(
                    safe_text(item["term"]),
                    styles["table_text"],
                ),
                Paragraph(
                    f"{float(item['contribution']):.4f}",
                    styles["table_text"],
                ),
            ]
        )

    table = Table(
        rows,
        colWidths=[
            11.5 * cm,
            4.85 * cm,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    INDIGO,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.75,
                    BORDER,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        LIGHT_BACKGROUND,
                    ],
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    return table


def build_recommendation_summary(
    best_match,
    styles,
):
    """
    Build the recommended-role summary card.
    """

    recommendation_data = [
        [
            Paragraph(
                "RECOMMENDED ROLE",
                styles["label"],
            ),
            Paragraph(
                "MATCH SCORE",
                styles["label"],
            ),
            Paragraph(
                "RECOMMENDATION",
                styles["label"],
            ),
        ],
        [
            Paragraph(
                safe_text(best_match["title"]),
                styles["value"],
            ),
            Paragraph(
                format_percentage(
                    best_match["score"]
                ),
                styles["value"],
            ),
            Paragraph(
                safe_text(
                    best_match[
                        "recommendation_level"
                    ]
                ),
                styles["value"],
            ),
        ],
    ]

    table = Table(
        recommendation_data,
        colWidths=[
            8.5 * cm,
            3.8 * cm,
            4.05 * cm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_INDIGO,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    INDIGO,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#C7D2FE"),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ]
        )
    )

    return table


def build_skills_table(
    best_match,
    styles,
):
    """
    Build matching and missing skill columns.
    """

    matching_skills = best_match.get(
        "matching_skills",
        [],
    )

    missing_skills = best_match.get(
        "missing_skills",
        [],
    )

    matching_text = (
        "<br/>".join(
            f"✓ {safe_text(skill)}"
            for skill in matching_skills[:15]
        )
        if matching_skills
        else "No explicitly matching skills were found."
    )

    missing_text = (
        "<br/>".join(
            f"• {safe_text(skill)}"
            for skill in missing_skills[:15]
        )
        if missing_skills
        else "No missing skills were identified."
    )

    rows = [
        [
            Paragraph(
                "MATCHING SKILLS",
                styles["table_header"],
            ),
            Paragraph(
                "SKILLS TO DEVELOP",
                styles["table_header"],
            ),
        ],
        [
            Paragraph(
                matching_text,
                styles["skill_text"],
            ),
            Paragraph(
                missing_text,
                styles["skill_text"],
            ),
        ],
    ]

    table = Table(
        rows,
        colWidths=[
            8.175 * cm,
            8.175 * cm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, 0),
                    SUCCESS,
                ),
                (
                    "BACKGROUND",
                    (1, 0),
                    (1, 0),
                    WARNING,
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (0, 1),
                    SUCCESS_BACKGROUND,
                ),
                (
                    "BACKGROUND",
                    (1, 1),
                    (1, 1),
                    WARNING_BACKGROUND,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.75,
                    BORDER,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    11,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    11,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
            ]
        )
    )

    return table


# --------------------------------------------------
# PDF Generation
# --------------------------------------------------

def create_pdf(report_data):
    """
    Generate a structured PDF career-analysis report.
    """

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2.1 * cm,
        leftMargin=2.1 * cm,
        topMargin=1.75 * cm,
        bottomMargin=1.65 * cm,
        title="AI Career Advisor — Career Analysis Report",
        author="Sara Hodzic and Mithat Misirlic",
        subject="AI-assisted resume and career analysis",
    )

    styles = build_styles()

    analysis = report_data["analysis"]
    best_match = report_data["best_match"]
    generated_at = report_data["generated_at"]

    story = []

    # --------------------------------------------------
    # Cover Header
    # --------------------------------------------------

    story.append(
        Spacer(
            1,
            0.55 * cm,
        )
    )

    story.append(
        Paragraph(
            "AI Career Advisor",
            styles["document_title"],
        )
    )

    story.append(
        Paragraph(
            (
                "Official Career Analysis Report"
                f"<br/>Generated "
                f"{generated_at.strftime('%d %B %Y at %H:%M')}"
            ),
            styles["document_subtitle"],
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=2.2,
            color=INDIGO,
            spaceAfter=0.55 * cm,
        )
    )

    # --------------------------------------------------
    # Executive Summary
    # --------------------------------------------------

    story.extend(
        section_heading(
            "1. Executive Summary",
            styles,
        )
    )

    story.append(
        build_summary_cards(
            analysis,
            styles,
        )
    )

    story.append(
        Spacer(
            1,
            0.45 * cm,
        )
    )

    story.append(
        build_confidence_box(
            analysis,
            styles,
        )
    )

    # --------------------------------------------------
    # Classification Results
    # --------------------------------------------------

    story.extend(
        section_heading(
            "2. Classification Results",
            styles,
        )
    )

    story.append(
        Paragraph(
            (
                "The machine-learning classifier evaluated "
                "the resume against the professional categories "
                "contained in the training dataset. The three "
                "highest-scoring categories are presented below."
            ),
            styles["body_muted"],
        )
    )

    story.append(
        Spacer(
            1,
            0.25 * cm,
        )
    )

    story.append(
        build_prediction_table(
            report_data["top_predictions"],
            styles,
        )
    )

    # --------------------------------------------------
    # Explainability
    # --------------------------------------------------

    story.extend(
        section_heading(
            "3. Prediction Explanation",
            styles,
        )
    )

    story.append(
        Paragraph(
            (
                "The following resume terms contributed most "
                "strongly to the primary category prediction. "
                "Higher contribution values indicate greater "
                "influence on the classifier's decision."
            ),
            styles["body_muted"],
        )
    )

    story.append(
        Spacer(
            1,
            0.25 * cm,
        )
    )

    if report_data["important_terms"]:

        story.append(
            build_feature_table(
                report_data["important_terms"],
                styles,
            )
        )

    else:

        story.append(
            Paragraph(
                "No influential terms were available.",
                styles["body"],
            )
        )

    # --------------------------------------------------
    # Job Recommendation
    # --------------------------------------------------

    story.extend(
        section_heading(
            "4. Best Job Recommendation",
            styles,
        )
    )

    story.append(
        build_recommendation_summary(
            best_match,
            styles,
        )
    )

    story.append(
        Spacer(
            1,
            0.4 * cm,
        )
    )

    story.append(
        build_skills_table(
            best_match,
            styles,
        )
    )

    # --------------------------------------------------
    # Interpretation and Limitations
    # --------------------------------------------------

    story.extend(
        section_heading(
            "5. Interpretation and Limitations",
            styles,
        )
    )

    story.append(
        Paragraph(
            (
                "This report provides decision support rather "
                "than a definitive professional assessment. "
                "Classification results depend on the language, "
                "structure and completeness of the submitted "
                "resume, as well as the categories represented "
                "in the model's training dataset."
            ),
            styles["body"],
        )
    )

    story.append(
        Spacer(
            1,
            0.25 * cm,
        )
    )

    story.append(
        Paragraph(
            (
                "Job recommendations and missing-skill results "
                "are based primarily on explicit skill matching. "
                "They may not capture transferable skills, "
                "synonyms, professional experience or deeper "
                "semantic relationships."
            ),
            styles["body"],
        )
    )

    # --------------------------------------------------
    # Document Closing
    # --------------------------------------------------

    story.append(
        Spacer(
            1,
            0.7 * cm,
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.7,
            color=BORDER,
            spaceAfter=0.35 * cm,
        )
    )

    story.append(
        Paragraph(
            (
                "<b>Generated by AI Career Advisor</b><br/>"
                "Sara Hodzic and Mithat Misirlic<br/>"
                "SRH University of Applied Sciences · 2026"
            ),
            styles["footer_note"],
        )
    )

    document.build(
        story,
        onFirstPage=draw_page_frame,
        onLaterPages=draw_page_frame,
    )

    buffer.seek(0)

    return buffer