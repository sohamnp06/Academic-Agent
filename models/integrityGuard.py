CHEAT_KEYWORDS = [
    "give answer",
    "exact answer",
    "solution",
    "solve this",
    "answer this",
    "question paper",
    "exam answer",
    "assignment answer",
    "copy paste",
    "direct answer"
]


def violates_integrity(query: str) -> bool:

    q = query.lower()

    for k in CHEAT_KEYWORDS:
        if k in q:
            return True

    return False


def integrity_response():

    return (
        "I can't help with direct answers, "
        "but I'm happy to explain the concept or walk you through the steps."
    )
