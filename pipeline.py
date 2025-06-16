import os
import re
import json
import logging
from markupsafe import Markup
from dotenv import load_dotenv
from models import Session, Data
from sklearn.cluster import KMeans
from langchain_groq import ChatGroq
from flask_login import current_user
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from sentence_transformers import SentenceTransformer
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.document_loaders import (PyPDFLoader, TextLoader, Docx2txtLoader)

load_dotenv()

groq_api_key = os.getenv('lang-graph-api')

# code block for printing colorful logging
LOG_COLORS = {
    'DEBUG': '\033[94m',   # Blue
    'INFO': '\033[92m',    # Green
    'WARNING': '\033[93m', # Yellow
    'ERROR': '\033[91m',   # Red
    'CRITICAL': '\033[95m' # Magenta
}
RESET = '\033[0m'

class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, RESET)
        record.levelname = f"{log_color}{record.levelname}{RESET}"
        record.msg = f"{log_color}{record.msg}{RESET}"
        return super().format(record)

handler = logging.StreamHandler()
formatter = ColorFormatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# llm model
llm_model = ChatGroq(
    api_key=groq_api_key,
    temperature=0.3,
    model="Llama3-8b-8192"
)

# embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# function to extract data from (.pdf, .docx, .txt)
def extract_text_from_doc(document: str) -> list[Document]:

    textual_data: list = []

    try:
        file_loaders = {
            ".txt":  TextLoader,
            ".docx": Docx2txtLoader,
            ".pdf":  PyPDFLoader,
        }

        file_extension = os.path.splitext(document)[1]
        loader_class = file_loaders.get(file_extension)

        if not loader_class:
            logging.error(f"Unsupported textual file type: {document}")

        loader = loader_class(document)
        loaded_data = loader.load()

        textual_data.extend(loaded_data)

    except Exception as e:
        logging.error(e)

    return textual_data

VALID_LABELS = {"Technical", "Experience", "Domain Knowledge", "Soft Skills"}

# function to normalize the labels
def normalize_label(label: str) -> str:
    label = label.strip().title()
    return label if label in VALID_LABELS else "Other"

# function to clean and extract data
def extract_gap_lines(summary: str) -> list[str]:
    cleaned_summary = re.sub(r'(<br\s*/?>)+', '\n', summary, flags=re.IGNORECASE)
    cleaned_summary = re.sub(r'<.*?>', '', cleaned_summary, flags=re.IGNORECASE)

    sections = re.findall(
        r"(SUMMARY OF FIT|GAP SUMMARY|KEY GAPS IDENTIFIED|GAPS & MISSING REQUIREMENTS|ACTIONABLE RECOMMENDATIONS|DETAILED OBSERVATIONS|RECOMMENDATIONS|REJECTION RISK)\n(.*?)(?=\n[A-Z\s]+(?:\n|$)|\Z)",
        cleaned_summary, flags=re.DOTALL | re.IGNORECASE
    )

    combined_text = "\n".join([s[1] for s in sections])
    lines = [line.strip("- ").strip() for line in combined_text.split("\n") if line.strip()]

    return lines

# function to create embeddings
def embed_gaps(gap_lines: list[str]):
    return embedding_model.encode(gap_lines)

# function to create clusters
def cluster_gaps_kmeans(gap_lines: list[str], num_clusters=4):
    embeddings = embed_gaps(gap_lines)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    clusters = {i: [] for i in range(num_clusters)}
    for label, line in zip(labels, gap_lines):
        clusters[label].append(line)
    return clusters

# function to generate labels
def label_cluster_with_llm(cluster_lines: list[str]) -> str:
    gap_score_prompt = PromptTemplate(
        input_variables=["cluster_data", "labels"],
        template="""
        You are an expert career advisor and hiring consultant.

        Your task is to analyze a job skill gaps. Based on the following labels, classify them into ONE of the following categories only.

        Below are the labels:
        ---
        {labels}
        ---

        Below is the cluster's data:
        ---
        {cluster_data}
        ---

        Return ONLY one label from: {labels}.
        No explanation. No formatting. Just the category name.
        """
    )

    filled_prompt = gap_score_prompt.format(
        cluster_data=cluster_lines,
        labels=VALID_LABELS
    )

    raw_output = llm_model.invoke(filled_prompt)

    return raw_output.content.strip().split("\n")[0].strip().title()

# function to generate score
def score_labeled_clusters(labeled_clusters: dict) -> dict:
    full_score_dict = {label: 0 for label in VALID_LABELS}

    max_count = max((len(v) for v in labeled_clusters.values()), default=1)

    for label in VALID_LABELS:
        count = len(labeled_clusters.get(label, []))
        score = round((count / max_count) * 100) if max_count else 0
        full_score_dict[label] = score

    return full_score_dict

# function to generate gap match score
def generate_gap_score(summary_text: str) -> dict | bool:
    try:
        gap_lines = extract_gap_lines(summary_text)
        if not gap_lines:
            return {}

        clustered = cluster_gaps_kmeans(gap_lines)
        labeled_clusters = {}
        for _, cluster_lines in clustered.items():
            raw_label = label_cluster_with_llm(cluster_lines)
            label = normalize_label(raw_label)
            if label not in labeled_clusters:
                labeled_clusters[label] = []
            labeled_clusters[label].extend(cluster_lines)

        return score_labeled_clusters(labeled_clusters)
    except Exception as e:
        logging.error(e)
        return False

# function to generate gap summary of user
def generate_gap_summary(parsed_data: str, job_desc: str) -> str|bool:

    try:
        topic_extraction_prompt = PromptTemplate(
            input_variables=["job_description", "resume_data"],
            template=(
                """
                You are an expert career advisor and hiring consultant.

                Your task is to analyze a candidate's resume against a job description for a specific role. Identify missing qualifications, skills, or experiences in the resume based on the job requirements, and provide a summary of the gaps along with actionable recommendations.

                Below is the job description:
                ---
                {job_description}
                ---

                Below is the candidate’s resume:
                ---
                {resume_data}
                ---

                Please return the following structured output:

                1. Summary of Fit:
                   - Briefly state how well the resume aligns with the job (e.g., strong, moderate, weak fit).
                   - Mention any relevant strengths the candidate has.

                2. Gaps & Missing Requirements:
                   - List all major skills, tools, qualifications, or experiences mentioned in the job description but missing or weak in the resume.
                   - Be specific: mention exact tool names, years of experience, project types, or domain knowledge that is absent or insufficient.

                3. Gap Summary:
                   - List the specific skills, experiences, tools, or qualifications mentioned in the job description that are not found or not clearly demonstrated in the resume.
                   - Clearly separate missing technical skills, domain knowledge, soft skills, and experience-related gaps.

                4. Key Gaps Identified
                    - List each missing requirement from the job description that isn't in the resume
                    - For each gap, specify:
                      * The exact requirement from the job description
                      * Evidence of its absence from the resume
                      * Importance level (critical, important, nice-to-have)

                5. Actionable Recommendations
                    - Provide specific suggestions to address each major gap:
                      * Skills to develop (with specific technologies/tools)
                      * Experience to gain (types of projects or roles)
                      * Resume improvements (what to add/emphasize/remove)
                      * Courses, certifications, or learning resources
                    - Prioritize recommendations by impact on job fit

                6. Detailed Observations:
                   - Identify any mismatches or partial matches (e.g., skill mentioned but not at required depth or experience level).
                   - Highlight any outdated technologies or irrelevant content taking up space.
                   - Note any partial matches or adjacent skills
                   - Point out irrelevant information that could be removed

                7. Recommendations:
                   - Suggest what the candidate should learn, improve, or highlight to better match this job.
                   - Recommend potential certifications, projects, or courses.
                   - Suggest how to rewrite or enhance certain parts of the resume to align better.

                8. Rejection Risk:
                   - State whether the resume would likely be rejected for this role and why.
                   - Include advice on how to bypass rejection in future iterations of the resume.

                IMPORTANT FORMATTING RULES:
                - Absolutely NO markdown symbols (*, **, +, etc.)
                - Section headers in ALL CAPS only
                - Use hyphen (-) for lists ONLY when specified above
                - Empty line between sections
                - Plain text only
                """
            )
        )

        filled_prompt = topic_extraction_prompt.format(
            job_description=job_desc,
            resume_data=parsed_data
        )
        raw_output = llm_model.invoke(filled_prompt)

        clean_prompt = """
        Remove ALL markdown formatting from the following text, including:
        - Asterisks (*) and double asterisks (**)
        - Plus signs (+)
        - Any other special formatting characters
        Preserve the content structure but use ONLY plain text formatting.
        Return the cleaned version:
        """ + raw_output.content

        cleaned_output = llm_model.invoke(clean_prompt)
        final_output = Markup(cleaned_output.content.strip().replace("\n", "<br>"))
        return final_output

    except Exception as e:
        logging.error(e)
        return False


# function to start mock interview with model
def mock_interview(query: str, difficulty: str) -> str:
    session = Session()

    user_data = (
        session.query(Data)
        .filter_by(user_id=current_user.id)
        .order_by(Data.id.desc())
        .first()
    )

    resume_text = user_data.resume_data
    job_description = user_data.job_description

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a technical interviewer for an AI/ML Lead role at a cybersecurity-focused company.

        Below is the job description:
        ---
        {job_description}
        ---
        
        Below is the candidate's resume:
        ---
        {resume_text}
        ---
        
        Below is the difficulty level:
        ---
        {difficulty}
        ---
        
        Using this information, generate a mock interview. The interview should consist of:
        
        1. 3 Behavioral Questions based on leadership, collaboration, and self-direction.
        2. 4 Technical Questions based on the required AI/ML pipeline experience (data processing, model training, MLOps).
        3. 2 Domain-Specific Questions related to cybersecurity or network traffic analytics.
        4. 1 Bonus or edge-case question testing adaptability, research, or latest trends in AI/ML.
        
        Format:
        - Question #
        - Type: [Behavioral/Technical/Domain-Specific/Bonus]
        - Question Text
        - Optional: Hints or what a strong answer would include.
        
        Keep the tone professional and focused. Ask questions tailored to the candidate’s background, projects, and any gaps or mismatches between the resume and the job description."""),

        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm_model

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: FileChatMessageHistory(f"instance/history_{current_user.id}.json"),
        input_messages_key="input",
        history_messages_key="history"
    )

    session_idd = str(current_user.id)

    response = chain_with_history.invoke(
        {"input": query},
        config=RunnableConfig(configurable={"session_id": session_idd})
    )

    return response.content if hasattr(response, 'content') else str(response)
