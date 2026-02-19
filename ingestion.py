import os
import pickle
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_PATH = "data/raw_pdfs"
VECTOR_DB_PATH = "vector_db"
INDEX_PATH = os.path.join(VECTOR_DB_PATH, "index.faiss")
DOC_PATH = os.path.join(VECTOR_DB_PATH, "documents.pkl")


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


def load_existing_data():

    if os.path.exists(INDEX_PATH) and os.path.exists(DOC_PATH):
        print("Loading existing vector database...")

        index = faiss.read_index(INDEX_PATH)

        with open(DOC_PATH, "rb") as f:
            documents = pickle.load(f)

        existing_sources = set(doc["source"] for doc in documents)

        return index, documents, existing_sources

    print("No existing vector DB found. Creating new one.")

    return None, [], set()


def ingest_new_documents(existing_sources):

    documents = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower().endswith(".pdf") and file not in existing_sources:

                full_path = os.path.join(root, file)
                print(f"Processing NEW file: {full_path}")

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


def save_data(index, documents):

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(DOC_PATH, "wb") as f:
        pickle.dump(documents, f)


def main():

    model = SentenceTransformer(
        "sentence-transformers/static-retrieval-mrl-en-v1",
        device="cpu"
    )

    index, existing_docs, existing_sources = load_existing_data()

    new_docs = ingest_new_documents(existing_sources)

    if not new_docs:
        print("No new PDFs found.")
        return

    texts = [doc["text"] for doc in new_docs]

    print("Embedding new chunks...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    if index is None:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    all_documents = existing_docs + new_docs

    save_data(index, all_documents)

    print("Vector DB updated!")
    print(f"Total chunks stored: {len(all_documents)}")


if __name__ == "__main__":
    main()
