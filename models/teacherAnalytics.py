import json

DB_PATH = "data/student_db.json"


def get_students_at_risk():

    with open(DB_PATH, "r") as f:
        students = json.load(f)

    results = []
    topic_counter = {}

    for student, info in students.items():

        mastery = info.get("topics", {})
        misconceptions = info.get("misconceptions", {})

        for topic, score in mastery.items():

            if score < 0.5:

                if score < 0.2:
                    status = "CRITICAL"
                    recommendation = "Immediate mentoring required"
                else:
                    status = "AT RISK"
                    recommendation = "Extra practice recommended"

                severity = round((0.5 - score) * 2, 2)

                topic_counter[topic] = topic_counter.get(topic, 0) + 1

                results.append({
                    "student": student,
                    "topic": topic,
                    "score": round(score, 2),
                    "status": status,
                    "severity": severity,
                    "recommendation": recommendation,
                    "misconceptions": misconceptions.get(topic, [])
                })

    topic_alerts = [t for t, c in topic_counter.items() if c >= 2]

    return results, topic_alerts
