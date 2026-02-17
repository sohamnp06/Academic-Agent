from models.llm import generate_answer

def detect_misconception(context_chunks, question, answer):

    context = "\n".join(context_chunks[:2])

    prompt = f"""
Student question:
{question}

Answer:
{answer}

Student still did not understand.

In ONE sentence identify the conceptual misunderstanding.
"""

    return generate_answer(context_chunks, prompt).strip()
