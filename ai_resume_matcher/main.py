#PYTHON CODE FOR AI RESUME MATCHER MAIN MODULE

from email.mime import base
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from docx import Document
import pdfplumber
import os
import uuid
import fitz  # PyMuPDF
import docx
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np  
from sqlalchemy import create_engine, Column, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

app = FastAPI(title="AI Resume Matcher - Ingestion Module")

# Create upload directories
RESUME_DIR = "uploads/resumes"
JD_DIR = "uploads/job_descriptions"

os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(JD_DIR, exist_ok=True)


# ---------- Utility Functions ----------

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    pdf = fitz.open(file_path)
    for page in pdf:
        text += page.get_text()
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()


def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return ""
import pdfplumber

def extract_pdf_text(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

from pypdf import PdfReader

def extract_pdf_text(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# ---------- Resume Upload API ----------

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1].lower()

    if file_ext not in ["pdf", "docx"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF and DOCX files are allowed"}
        )

    resume_id = str(uuid.uuid4())
    file_path = f"{RESUME_DIR}/{resume_id}.{file_ext}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    raw_text = extract_text(file_path)
    processed_text = preprocess_text(raw_text)
    embedding=generate_embedding(processed_text)

    db=SessionLocal()

    resume=Resume(
        resume_id=resume_id,
        filename=file.filename,
        processed_text=processed_text,
        embedding=json.dumps(embedding.tolist())
    )
    
    
    db.add(resume)
    db.commit()
    db.close()


# ---------- Job Description Upload API ----------

@app.post("/upload-job-description")
async def upload_job_description(
    jd_text: str = Form(None),
    file: UploadFile = File(None)
):
    jd_id = str(uuid.uuid4())
    
    # Case 1: JD entered as text
   
    
    if jd_text:
        raw_text=jd_text.strip() 
        processed_text=preprocess_text(raw_text)
        embedding=generate_embedding(processed_text)  
        db=SessionLocal()
        jd=JobDescription(
            id=jd_id,
            processed_text=processed_text,
            embedding=json.dumps(embedding.tolist())
        )
        db.add(jd)
        db.commit()
        db.close()
        return {
            "job_description_id": jd_id,
            "source": "text",
            "raw_text": raw_text,
            "embedding": embedding.tolist(),
            "processed_text": processed_text
        }
    # Case 2: JD uploaded as file
    if file:
        
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in ["pdf", "docx"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Only PDF and DOCX files allowed"}
            )

        file_path = f"{JD_DIR}/{jd_id}.{file_ext}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        raw_text = extract_text(file_path)
        processed_text = preprocess_text(raw_text)
        embedding = generate_embedding(processed_text)
        
        return {
            "job_description_id": jd_id,
            "source": "file",
            "raw_text": raw_text,
            "embedding": embedding.tolist(),
            "processed_text": processed_text
        }

    return JSONResponse(
        status_code=400,
        content={"error": "Provide either JD text or a file"}
    )

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


@app.post("/match-resume")
async def match_resume(
    resume_embedding: list = Form(...),
    jd_embedding: list = Form(...)
):
    resume_vec = np.array(resume_embedding)
    jd_vec = np.array(jd_embedding)

    match_percentage = calculate_similarity(resume_vec, jd_vec)

    return {
        "match_percentage": match_percentage
    }



# ---------- STEP 4: TEXT PREPROCESSING ----------

def preprocess_text(text: str) -> str:
    # 1. Lowercase
    text = text.lower()

    # 2. Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords & short words
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    # 5. Join back to string
    processed_text = " ".join(cleaned_tokens)

    return processed_text

model=SentenceTransformer('all-MiniLM-L6-v2')  # Example model

def generate_embedding(text: str):
    embedding = model.encode(text)
    return embedding

def calculate_similarity(resume_embedding, jd_embedding):
    similarity = cosine_similarity(
        [resume_embedding],
        [jd_embedding]
    )[0][0]

    return round(similarity * 100, 2)  # percentage

def rank_resumes(resume_embeddings, resume_ids, jd_embedding):
    """
    resume_embeddings: list of numpy arrays
    resume_ids: list of resume IDs
    jd_embedding: numpy array
    """

    scores = []

    for i in range(len(resume_embeddings)):
        similarity = cosine_similarity(
            [resume_embeddings[i]],
            [jd_embedding]
        )[0][0]

        scores.append({
            "resume_id": resume_ids[i],
            "match_percentage": round(similarity * 100, 2)
        })

    # Sort resumes by match percentage (descending)
    ranked_resumes = sorted(
        scores,
        key=lambda x: x["match_percentage"],
        reverse=True
    )

    return ranked_resumes



# ---------- STEP 6 (MVP): TF-IDF MODEL ----------

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

def tfidf_match(resume_text: str, jd_text: str) -> float:
    vectors = tfidf_vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)
## ---------- STEP 7 (ADVANCED): SBERT MODEL ----------##

def sbert_match(resume_embedding, jd_embedding) -> float:
    """
    Computes similarity using Sentence-BERT embeddings
    """
    similarity = cosine_similarity(
        [resume_embedding],
        [jd_embedding]
    )[0][0]

    return round(similarity * 100, 2)

##--select model API--##


@app.post("/match")
async def match_resume(
    resume_text: str = Form(...),
    jd_text: str = Form(...),
    resume_embedding: list = Form(None),
    jd_embedding: list = Form(None),
    model_type: str = Form("sbert")
):
    """
    model_type:
    - 'tfidf' (MVP)
    - 'sbert' (Advanced)
    """

    if model_type == "tfidf":
        score = tfidf_match(resume_text, jd_text)

    elif model_type == "sbert":
        score = sbert_match(
            np.array(resume_embedding),
            np.array(jd_embedding)
        )

    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid model type"}
        )

    return {
        "model_used": model_type,
        "match_percentage": score
    }
##----ranking resume----##
@app.post("/rank-resumes")
async def rank_all_resumes(
    resume_ids: list = Form(...),
    resume_embeddings: list = Form(...),
    jd_embedding: list = Form(...)
):
    """
    resume_ids: list of resume IDs
    resume_embeddings: list of resume embeddings
    jd_embedding: job description embedding
    """

    # Convert to numpy
    resume_embeddings_np = [np.array(e) for e in resume_embeddings]
    jd_embedding_np = np.array(jd_embedding)

    ranked_resumes = rank_resumes(
        resume_embeddings_np,
        resume_ids,
        jd_embedding_np
    )

    return {
        "ranked_resumes": ranked_resumes
    }
    db = SessionLocal()

    for result in ranked_resumes:
        match = MatchResult(
            id=str(uuid.uuid4()),
            resume_id=result["resume_id"],
            jd_id=jd_id,
            score=result["match_percentage"]
        )
        db.add(match)

    db.commit()
    db.close()

# ---------- STEP 8: DATABASE SETUP ----------

DATABASE_URL = "sqlite:///resume_matcher.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(String, primary_key=True)
    filename = Column(String)
    processed_text = Column(Text)
    embedding = Column(Text)  # stored as JSON

class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(String, primary_key=True)
    processed_text = Column(Text)
    embedding = Column(Text)

class MatchResult(Base):
    __tablename__ = "match_results"

    id = Column(String, primary_key=True)
    resume_id = Column(String)
    jd_id = Column(String)
    score = Column(Float)

Base.metadata.create_all(engine)
