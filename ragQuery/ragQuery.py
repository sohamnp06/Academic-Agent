import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

from models.topicMapper import extract_topic
from models.studentModel import update_mastery, add_misconception, get_misconceptions
from models.misconceptionDetector import detect_misconception
from models.adaptiveAnswer import generate_adaptive_answer
from models.teacherAnalytics import get_students_at_risk
from models.integrityGuard import violates_integrity, integrity_response

VECTOR_DB_PATH = "vector_db"

index = faiss.read_index(f"{VECTOR_DB_PATH}/index.faiss")

with open(f"{VECTOR_DB_PATH}/documents.pkl", "rb") as f:
    documents = pickle.load(f)
    
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")



def retrieve_context(query, top_k=3):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(documents[idx]["text"])

    return results


if __name__ == "__main__":

    print("\nAcademic Agent Ready (type exit to quit)\n")

    while True:

        query = input("Ask: ").strip()

        if query.lower() == "exit":
            break

        if violates_integrity(query):
            print("\n--- Integrity Notice ---\n")
            print(integrity_response())
            print("\n------------------------\n")
            continue

        if query.lower() == "teacher":

            risks, topic_alerts = get_students_at_risk()

            print("\n===== TEACHER DASHBOARD =====\n")

            if not risks:
                print("No students currently at risk.\n")

            for r in risks:

                print(f"Student: {r['student']}")
                print(f"Topic: {r['topic']}")
                print(f"Mastery: {r['score']}")
                print(f"Status: {r['status']}")
                print(f"Recommendation: {r['recommendation']}")

                if r["misconceptions"]:
                    print("Misconceptions:")
                    for m in r["misconceptions"]:
                        print("-", m)
                else:
                    print("Misconceptions: None")

                print("\n-------------------------\n")

            if topic_alerts:
                print("Topics needing curriculum review:")
                for t in topic_alerts:
                    print("-", t)

            print("\n    END OF DASHBOARD \n")
            continue

        student_id = "akash"

        topic = extract_topic(query)

        chunks = retrieve_context(query)

        misconceptions = get_misconceptions(student_id, topic)
        answer = generate_adaptive_answer(chunks, query, misconceptions)

        print("\n--- Answer ---\n")
        print(answer)
        print("\n------------\n")

        student_feedback = input("Did you understand? (yes/no): ").lower()

        if student_feedback == "yes":
            interaction_quality = 1
        else:
            interaction_quality = -1
           
            misconception = detect_misconception(chunks, query, answer)
            add_misconception(student_id, topic, misconception)
            print(f"\nDetected Misconception: {misconception}\n")

        new_score = update_mastery(student_id, topic, interaction_quality)

        print(f"\nUpdated mastery for {topic}: {round(new_score, 2)}\n")
