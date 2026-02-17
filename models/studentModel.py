import json
import os

DB_FILE = "data/student_db.json"


def load_students():
    if not os.path.exists(DB_FILE):
        return {}

    with open(DB_FILE, "r") as f:
        return json.load(f)


def save_students(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)


def update_mastery(student_id, topic, interaction_quality):

    students = load_students()

    if student_id not in students:
        students[student_id] = {
            "topics": {},
            "misconceptions": {}
        }

    if "misconceptions" not in students[student_id]:
        students[student_id]["misconceptions"] = {}

    topics = students[student_id]["topics"]

    if topic not in topics:
        topics[topic] = 0.5

    learning_rate = 0.15

    topics[topic] = topics.get(topic, 0.5) * 0.9 + learning_rate * interaction_quality
    topics[topic] = max(0.0, min(1.0, topics[topic]))

    save_students(students)

    return topics[topic]


def add_misconception(student_id, topic, misconception):

    students = load_students()

    if student_id not in students:
        students[student_id] = {
            "topics": {},
            "misconceptions": {}
        }

    if topic not in students[student_id]["misconceptions"]:
        students[student_id]["misconceptions"][topic] = []

    students[student_id]["misconceptions"][topic].append(misconception)

    save_students(students)


def get_misconceptions(student_id, topic):

    students = load_students()

    if student_id not in students:
        return []

    return students[student_id].get("misconceptions", {}).get(topic, [])
