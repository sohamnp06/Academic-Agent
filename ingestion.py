import os
import pickle
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

DATA_PATH = "data/raw_pdfs"
VECTOR_DB_PATH = "vector_db"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    return splitter.split_text(text)


def ingest_documents():
    documents = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                print(f"Processing: {full_path}")

                text = extract_text_from_pdf(full_path)

                if len(text.strip()) < 50:
                    print("Skipping empty or scanned document.")
                    continue

                chunks = chunk_text(text)

                for chunk in chunks:
                    documents.append({
                        "text": chunk,
                        "source": file
                    })

    return documents


def create_vector_store(documents):

    if len(documents) == 0:
        print("No valid text documents found.")
        return

    print("Creating embeddings on:", DEVICE)

    model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    texts = [doc["text"] for doc in documents]

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    faiss.write_index(index, os.path.join(VECTOR_DB_PATH, "index.faiss"))

    with open(os.path.join(VECTOR_DB_PATH, "documents.pkl"), "wb") as f:
        pickle.dump(documents, f)

    print("Vector database created successfully!")


if __name__ == "__main__":
    print("Starting ingestion...")
    docs = ingest_documents()
    print(f"Total chunks created: {len(docs)}")
    create_vector_store(docs)
