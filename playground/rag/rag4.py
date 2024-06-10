# type: ignore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, RagRetriever
import os

# temporary fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-uncased")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google-bert/bert-base-multilingual-cased", retriever=retriever)

input_dict = tokenizer.prepare_seq2seq_batch("who holds the record in 100m freestyle", return_tensors="pt")

generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
