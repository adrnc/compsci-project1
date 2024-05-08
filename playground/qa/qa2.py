from transformers import pipeline

model_checkpoint = "consciousAI/question-answering-roberta-base-s-v2"

context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"

question_answerer = pipeline("question-answering", model=model_checkpoint)

answer = question_answerer(question=question, context=context)
print(answer)
