from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/static-retrieval-mrl-en-v1",
    device="cpu"
)

TOPICS = ["maths", "coding", "general", "sports"]

topic_embeddings = model.encode(TOPICS, normalize_embeddings=True)


def extract_topic(question):

    q_embed = model.encode([question], normalize_embeddings=True)

    scores = np.dot(q_embed, topic_embeddings.T)

    return TOPICS[np.argmax(scores)]
