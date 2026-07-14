from pypdf import PdfReader


def load_resume(uploaded_file, resume_df, resume_index):
    """
    Load a resume from an uploaded PDF/TXT file or the bundled dataset.

    Returns:
        tuple[str, str]: Resume text and actual category label.
    """

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            pdf = PdfReader(uploaded_file)
            resume_text = ""

            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text:
                    resume_text += page_text + "\n"
        else:
            resume_text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore",
            )

        actual_category = "User Uploaded Resume"
    else:
        resume = resume_df.iloc[resume_index]
        resume_text = resume["Resume_str"]
        actual_category = resume["Category"]

    return resume_text, actual_category
