from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# TODO: https://huggingface.co/facebook/rag-token-nq
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)

generator = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")

    with tokenizer.as_target_tokenizer():
        outputs = generator(**inputs)

    predicted_answer = tokenizer.decode(outputs["generated_tokens"][0], skip_special_tokens=True)
    return predicted_answer

question = "What is RAG?"
context = "RAG (Retrieval-Augmented Generation) is a model that combines retrieval-based and generation-based approaches for question answering."

answer = answer_question(question, context)
print("Answer:", answer)
