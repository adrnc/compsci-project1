# type: ignore
from transformers import pipeline, set_seed

qa_model_name = "consciousAI/question-answering-roberta-base-s-v2"
chat_model_name = "openai-community/gpt2-xl"

chat_max_length = 256
chat_num_return_sequences = 1

context = "🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other."
question = "Which deep learning libraries back 🤗 Transformers?"

print(f"Context: {context}")
print(f"Question: {question}")

qa_model = pipeline("question-answering", model=qa_model_name)

def encode_chat_question(question: str, context: str):
    return f"Instruct: Write an answer to the question \"{question}\" including the words \"{context}\".\nOutput: "

def decode_chat_answer(text: str, prompt: str):
    return text[len(prompt):]

qa_result = qa_model(question=question, context=context)
qa_answer = qa_result["answer"]

print(f"\nQA: {qa_answer}")

chat_model = pipeline("text-generation", model=chat_model_name)

set_seed(42)

chat_prompt = encode_chat_question(question, qa_answer)
chat_results = chat_model(chat_prompt, max_length=chat_max_length, num_return_sequences=chat_num_return_sequences, pad_token_id=50256, truncation=False)

if not isinstance(chat_results, list):
    chat_results = [chat_results]

for chat_result in chat_results:
    chat_text = chat_result["generated_text"]
    print(f"Chat: {decode_chat_answer(chat_text, chat_prompt)}")
