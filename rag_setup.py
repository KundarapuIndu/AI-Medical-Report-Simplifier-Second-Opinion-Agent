from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MEDICAL_KNOWLEDGE = """
HbA1c: Normal below 5.7%. Prediabetes 5.7-6.4%. Diabetes 6.5% and above.
Source: American Diabetes Association guidelines 2023.

eGFR: Normal above 60 mL/min/1.73m2. Stage 3 CKD: 30-59. Stage 4: 15-29.
Source: KDIGO CKD Guidelines 2022.

TSH: Normal range 0.4-4.0 µIU/mL. Below 0.4 may indicate hyperthyroidism.
Source: American Thyroid Association 2023.

Haemoglobin: Normal male 13.5-17.5 g/dL. Female 12.0-15.5 g/dL.
Source: WHO Anaemia guidelines.
"""


def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore() -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents([MEDICAL_KNOWLEDGE])

    vectorstore = Chroma.from_documents(
        chunks, _embeddings(), persist_directory=CHROMA_DIR
    )
    print("Vectorstore built at", Path(CHROMA_DIR).resolve())
    return vectorstore


def get_vectorstore() -> Chroma:
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        return build_vectorstore()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings())
