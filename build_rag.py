"""Run once to build the ChromaDB knowledge base: python build_rag.py"""
from rag_setup import build_vectorstore

if __name__ == "__main__":
    build_vectorstore()
