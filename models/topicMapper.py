from sentence_transformers import SentenceTransformer
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

TOPICS = ["maths", "coding", "general", "sports"]

topic_embeddings = model.encode(TOPICS, normalize_embeddings=True)


def extract_topic(question):

    q_embed = model.encode([question], normalize_embeddings=True)

    scores = np.dot(q_embed, topic_embeddings.T)

    return TOPICS[np.argmax(scores)]
