# type: ignore
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import os

# temporary fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Load the tokenizer, retriever, and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# Define the question
question = "What is the capital of France?"

# Tokenize the input
inputs = tokenizer(question, return_tensors="pt")

# Generate the answer
generated = model.generate(
    input_ids=inputs["input_ids"],
    max_length=50,  # Adjust max_length to control the length of the generated answer
    num_beams=5,    # Use beam search to improve the quality of the answer
    early_stopping=True
)

# Decode the generated answer
answer = tokenizer.decode(generated[0], skip_special_tokens=True)

print(answer)
