import streamlit as st
from pypdf import PdfReader
import re
import csv
import math
from io import StringIO
from collections import Counter


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Screening System")

st.write(
    "Screen multiple resumes against a job description "
    "using skills, education, experience and NLP-based "
    "text similarity."
)


# =========================================================
# SKILLS DATABASE
# =========================================================

SKILLS = [
    "python",
    "java",
    "c++",
    "javascript",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "nlp",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "streamlit",
    "flask",
    "django",
    "git",
    "github",
    "docker",
    "html",
    "css",
    "react",
    "excel",
    "power bi",
    "tableau",
]


# =========================================================
# EDUCATION GROUPS
# =========================================================

EDUCATION_GROUPS = {

    "bachelor": [
        "b.tech",
        "btech",
        "b.e",
        "be",
        "b.sc",
        "bsc",
        "bca",
        "bachelor",
        "bachelors",
        "bachelor's",
        "bachelor degree",
        "bachelor's degree",
    ],

    "master": [
        "m.tech",
        "mtech",
        "m.e",
        "me",
        "m.sc",
        "msc",
        "mca",
        "mba",
        "master",
        "masters",
        "master's",
        "master degree",
        "master's degree",
    ],

    "phd": [
        "phd",
        "ph.d",
        "doctorate",
    ],

    "diploma": [
        "diploma",
        "d.pharm",
        "dpharm",
        "pharmacy diploma",
    ],
}


# =========================================================
# EXPERIENCE PATTERNS
# =========================================================

EXPERIENCE_PATTERNS = [
    r"\b\d+(?:\.\d+)?\+?\s*years?\s+of\s+experience\b",
    r"\b\d+(?:\.\d+)?\+?\s*years?\s+experience\b",
    r"\b\d+(?:\.\d+)?\+?\s*months?\s+of\s+experience\b",
    r"\b\d+(?:\.\d+)?\+?\s*months?\s+experience\b",
    r"\binternship\b",
    r"\bintern\b",
    r"\bwork experience\b",
    r"\bprofessional experience\b",
]


# =========================================================
# NLP STOP WORDS
# =========================================================

STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "from",
    "by",
    "as",
    "at",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "he",
    "she",
    "his",
    "her",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
    "should",
    "candidate",
    "candidates",
}


# =========================================================
# NLP - TOKENIZATION
# =========================================================

def tokenize_text(text):
    """
    Convert text into lowercase word tokens.
    Removes common stop words.
    """

    text = text.lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9+#.]+\b",
        text
    )

    filtered_tokens = []

    for token in tokens:

        if token not in STOP_WORDS:

            filtered_tokens.append(token)

    return filtered_tokens


# =========================================================
# NLP - TERM FREQUENCY
# =========================================================

def calculate_tf(tokens):
    """
    Calculate Term Frequency (TF).
    """

    word_counts = Counter(tokens)

    total_words = len(tokens)

    if total_words == 0:

        return {}

    tf = {}

    for word, count in word_counts.items():

        tf[word] = count / total_words

    return tf


# =========================================================
# NLP - INVERSE DOCUMENT FREQUENCY
# =========================================================

def calculate_idf(documents):
    """
    Calculate Inverse Document Frequency (IDF).
    """

    total_documents = len(documents)

    document_frequency = Counter()

    for document in documents:

        unique_words = set(document)

        for word in unique_words:

            document_frequency[word] += 1

    idf = {}

    for word, frequency in document_frequency.items():

        idf[word] = math.log(
            (1 + total_documents)
            / (1 + frequency)
        ) + 1

    return idf


# =========================================================
# NLP - TF-IDF
# =========================================================

def calculate_tfidf(tokens, idf):
    """
    Calculate TF-IDF vector.
    """

    tf = calculate_tf(tokens)

    tfidf = {}

    for word, tf_value in tf.items():

        tfidf[word] = (
            tf_value
            * idf.get(word, 0)
        )

    return tfidf


# =========================================================
# NLP - COSINE SIMILARITY
# =========================================================

def calculate_cosine_similarity(
    vector1,
    vector2
):
    """
    Calculate cosine similarity between
    two TF-IDF vectors.
    """

    common_words = (
        set(vector1.keys())
        & set(vector2.keys())
    )

    dot_product = sum(

        vector1[word]
        * vector2[word]

        for word in common_words
    )

    magnitude1 = math.sqrt(

        sum(
            value ** 2
            for value in vector1.values()
        )
    )

    magnitude2 = math.sqrt(

        sum(
            value ** 2
            for value in vector2.values()
        )
    )

    if magnitude1 == 0 or magnitude2 == 0:

        return 0

    similarity = (

        dot_product
        / (magnitude1 * magnitude2)

    )

    return similarity


# =========================================================
# NLP - FINAL SIMILARITY FUNCTION
# =========================================================

def calculate_nlp_similarity(
    job_description,
    resume_text
):
    """
    Calculate NLP-based similarity between
    job description and resume using
    TF-IDF and cosine similarity.
    """

    job_tokens = tokenize_text(
        job_description
    )

    resume_tokens = tokenize_text(
        resume_text
    )

    documents = [
        job_tokens,
        resume_tokens
    ]

    idf = calculate_idf(
        documents
    )

    job_vector = calculate_tfidf(
        job_tokens,
        idf
    )

    resume_vector = calculate_tfidf(
        resume_tokens,
        idf
    )

    similarity = calculate_cosine_similarity(
        job_vector,
        resume_vector
    )

    return similarity * 100


# =========================================================
# EXTRACT REQUIRED EXPERIENCE
# =========================================================

def extract_required_experience(text):

    text_lower = text.lower()

    years = 0
    months = 0

    year_matches = re.findall(
        r"(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of\s*)?experience",
        text_lower
    )

    for value in year_matches:

        years += float(value)

    month_matches = re.findall(
        r"(\d+(?:\.\d+)?)\+?\s*months?\s*(?:of\s*)?experience",
        text_lower
    )

    for value in month_matches:

        months += float(value)

    total_months = (
        years * 12
    ) + months

    return total_months


# =========================================================
# EXTRACT CANDIDATE EXPERIENCE
# =========================================================

def extract_candidate_experience(text):

    text_lower = text.lower()

    years = 0
    months = 0

    year_matches = re.findall(
        r"(\d+(?:\.\d+)?)\+?\s*years?\s*(?:of\s*)?experience",
        text_lower
    )

    for value in year_matches:

        years += float(value)

    month_matches = re.findall(
        r"(\d+(?:\.\d+)?)\+?\s*months?\s*(?:of\s*)?experience",
        text_lower
    )

    for value in month_matches:

        months += float(value)

    total_months = (
        years * 12
    ) + months

    return total_months


# =========================================================
# FIND SKILLS
# =========================================================

def find_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        if skill == "c++":

            pattern = (
                r"(?<!\w)c\+\+(?!\w)"
            )

        else:

            pattern = (
                r"(?<!\w)"
                + re.escape(skill)
                + r"(?!\w)"
            )

        if re.search(
            pattern,
            text
        ):

            found_skills.append(skill)

    return found_skills


# =========================================================
# FIND EDUCATION
# =========================================================

def find_education(text):

    text_lower = text.lower()

    found_groups = []

    for group, keywords in EDUCATION_GROUPS.items():

        for keyword in keywords:

            if keyword in ["be", "me"]:

                pattern = (
                    r"(?<![a-z])"
                    + re.escape(keyword)
                    + r"(?![a-z])"
                )

                if re.search(
                    pattern,
                    text_lower
                ):

                    found_groups.append(group)

                    break

            else:

                if keyword in text_lower:

                    found_groups.append(group)

                    break

    return list(
        dict.fromkeys(
            found_groups
        )
    )


# =========================================================
# FIND EXPERIENCE TEXT
# =========================================================

def find_experience(text):

    text_lower = text.lower()

    experience_matches = []

    for pattern in EXPERIENCE_PATTERNS:

        matches = re.findall(
            pattern,
            text_lower
        )

        experience_matches.extend(
            matches
        )

    return list(
        dict.fromkeys(
            experience_matches
        )
    )


# =========================================================
# JOB DESCRIPTION
# =========================================================

st.subheader(
    "1️⃣ Job Description"
)

job_description = st.text_area(

    "Enter the job description",

    height=200,

    placeholder=(
        "Example: We are looking for a Python Developer "
        "with knowledge of Python, SQL, Machine Learning, "
        "Pandas and NumPy. Bachelor's degree preferred. "
        "2 years of experience required."
    )
)


# =========================================================
# MULTIPLE RESUME UPLOAD
# =========================================================

st.subheader(
    "2️⃣ Upload Resumes"
)

uploaded_files = st.file_uploader(

    "Upload one or more Resume PDFs",

    type=["pdf"],

    accept_multiple_files=True
)


# =========================================================
# SCREENING
# =========================================================

if uploaded_files and job_description.strip():

    st.subheader(
        "3️⃣ Screening Resumes"
    )

    # -----------------------------------------------------
    # JOB REQUIREMENTS
    # -----------------------------------------------------

    job_skills = find_skills(
        job_description
    )

    job_education = find_education(
        job_description
    )

    job_experience = find_experience(
        job_description
    )

    required_experience_months = (
        extract_required_experience(
            job_description
        )
    )

    results = []


    # -----------------------------------------------------
    # PROCESS EACH RESUME
    # -----------------------------------------------------

    for uploaded_file in uploaded_files:

        reader = PdfReader(
            uploaded_file
        )

        resume_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:

                resume_text += (
                    text + "\n"
                )


        # -------------------------------------------------
        # RESUME INFORMATION
        # -------------------------------------------------

        resume_skills = find_skills(
            resume_text
        )

        resume_education = find_education(
            resume_text
        )

        resume_experience = find_experience(
            resume_text
        )

        candidate_experience_months = (
            extract_candidate_experience(
                resume_text
            )
        )


        # -------------------------------------------------
        # SKILL MATCHING
        # -------------------------------------------------

        matched_skills = [

            skill

            for skill in job_skills

            if skill in resume_skills

        ]

        missing_skills = [

            skill

            for skill in job_skills

            if skill not in resume_skills

        ]


        if job_skills:

            skill_score = (

                len(matched_skills)
                / len(job_skills)

            ) * 100

        else:

            skill_score = 100


        # -------------------------------------------------
        # EDUCATION MATCHING
        # -------------------------------------------------

        if job_education:

            education_match = any(

                education in resume_education

                for education
                in job_education

            )

            if education_match:

                education_score = 100

            else:

                education_score = 0

        else:

            education_score = 100


        # -------------------------------------------------
        # EXPERIENCE MATCHING
        # -------------------------------------------------

        if required_experience_months > 0:

            if candidate_experience_months > 0:

                experience_score = min(

                    (
                        candidate_experience_months
                        / required_experience_months
                    ) * 100,

                    100

                )

            else:

                experience_score = 0

        else:

            if job_experience:

                if resume_experience:

                    experience_score = 100

                else:

                    experience_score = 0

            else:

                experience_score = 100


        # -------------------------------------------------
        # STRUCTURED SCORE
        # -------------------------------------------------

        structured_score = (

            skill_score * 0.60

            + education_score * 0.20

            + experience_score * 0.20

        )


        # -------------------------------------------------
        # NLP SIMILARITY
        # -------------------------------------------------

        nlp_score = calculate_nlp_similarity(

            job_description,

            resume_text

        )


        # -------------------------------------------------
        # FINAL COMBINED SCORE
        # -------------------------------------------------

        final_score = (

            structured_score * 0.70

            + nlp_score * 0.30

        )


        # -------------------------------------------------
        # EXPERIENCE DISPLAY
        # -------------------------------------------------

        if candidate_experience_months > 0:

            years = (
                candidate_experience_months
                / 12
            )

            experience_display = (
                f"{years:.1f} years"
            )

        elif resume_experience:

            experience_display = ", ".join(
                resume_experience
            )

        else:

            experience_display = (
                "No experience detected"
            )


        # -------------------------------------------------
        # EDUCATION DISPLAY
        # -------------------------------------------------

        if resume_education:

            education_display = ", ".join(

                education.title()

                for education
                in resume_education

            )

        else:

            education_display = (
                "No education detected"
            )


        # -------------------------------------------------
        # STORE RESULT
        # -------------------------------------------------

        results.append({

            "Candidate":
                uploaded_file.name,

            "Skills":
                ", ".join(
                    matched_skills
                ),

            "Missing Skills":
                ", ".join(
                    missing_skills
                ),

            "Education":
                education_display,

            "Experience":
                experience_display,

            "NLP Similarity":
                round(
                    nlp_score,
                    1
                ),

            "Score":
                round(
                    final_score,
                    1
                )
        })


    # =====================================================
    # SORT RESULTS
    # =====================================================

    results = sorted(

        results,

        key=lambda x: x["Score"],

        reverse=True

    )


    # =====================================================
    # ADD POSITION NUMBER
    # =====================================================

    for index, result in enumerate(

        results,

        start=1

    ):

        result["No."] = index


    # =====================================================
    # RESULTS TABLE
    # =====================================================

    st.subheader(
        "📊 Screening Results"
    )

    display_results = []

    for result in results:

        display_results.append({

            "No.":
                result["No."],

            "Candidate":
                result["Candidate"],

            "Skills":
                result["Skills"],

            "Missing Skills":
                result["Missing Skills"],

            "Education":
                result["Education"],

            "Experience":
                result["Experience"],

            "NLP Similarity":
                result["NLP Similarity"],

            "Final Score":
                result["Score"]

        })


    st.dataframe(

        display_results,

        width="stretch"

    )


    # =====================================================
    # SCORE EXPLANATION
    # =====================================================

    st.info(
        "Final Score = 70% structured matching "
        "(skills, education and experience) + "
        "30% NLP similarity using TF-IDF and "
        "cosine similarity."
    )


    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    csv_buffer = StringIO()

    fieldnames = [

        "No.",
        "Candidate",
        "Skills",
        "Missing Skills",
        "Education",
        "Experience",
        "NLP Similarity",
        "Final Score"

    ]

    writer = csv.DictWriter(

        csv_buffer,

        fieldnames=fieldnames

    )

    writer.writeheader()

    writer.writerows(
        display_results
    )

    csv_data = (
        csv_buffer.getvalue()
    )


    st.download_button(

        label="⬇️ Download Screening Results",

        data=csv_data,

        file_name=(
            "resume_screening_results.csv"
        ),

        mime="text/csv"

    )


# =========================================================
# WARNING
# =========================================================

elif uploaded_files and not job_description.strip():

    st.warning(

        "⚠️ Please enter a job description "
        "before screening resumes."

    )