from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading local LLM...(first time may take ~30 seconds)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200
)


def generate_answer(context_chunks, question):

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an academic assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

If answer not found in context, say:
"I don't find this in the provided academic material."

Answer:
"""

    output = generator(prompt)[0]["generated_text"]

    return output.split("Answer:")[-1].strip()
