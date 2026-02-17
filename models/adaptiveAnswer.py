from models.llm import generate_answer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_adaptive_answer(context_chunks, question, misconceptions):

    context = "\n\n".join(context_chunks)

    misconception_text = ""

    if misconceptions:
        misconception_text = (
            "The student previously misunderstood:\n"
            + "\n".join(misconceptions)
            + "\n\nPlease simplify and explicitly correct these misunderstandings.\n"
        )

    prompt = f"""
You are an expert university tutor.

{misconception_text}

Student question:
{question}

Explain clearly with:
- simple language
- one example
- step by step reasoning

Context:
{context}
"""

    return generate_answer(context_chunks, prompt)
