import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import *

def ingest_pdf(file_name):
    # Get absolute path of current file (important)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_name)

    print("Loading PDF from:", file_path)

    # Check if file exists (good practice)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    print("Creating embeddings...")
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    print("💾 Storing in Chroma DB...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=os.path.join(base_dir, DB_DIR)
    )

    db.persist()
    print("Done!")

if __name__ == "__main__":
    ingest_pdf("sample.pdf")