from pypdf import PdfReader


def load_resume(uploaded_file, resume_df, resume_index):
    """
    Loads a resume from either:
    - uploaded PDF
    - uploaded TXT
    - dataset

    Returns:
        resume_text
        actual_category
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
                errors="ignore"
            )

        actual_category = "User Uploaded Resume"

    else:

        resume_text = resume_df.iloc[
            resume_index
        ]["Resume_str"]

        actual_category = resume_df.iloc[
            resume_index
        ]["Category"]

    return resume_text, actual_category