# type: ignore
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, RagConfig
import os

# temporary fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# docs: https://huggingface.co/docs/transformers/v4.40.2/en/model_doc/rag

pretrained_tokenizer = "facebook/rag-token-nq"
pretrained_retriever = "facebook/rag-token-nq"
pretrained_model = "facebook/rag-token-nq"

# the tokenizer transforms the input data into token
tokenizer = RagTokenizer.from_pretrained(pretrained_tokenizer)

# the retriever collects the documents
retriever = RagRetriever.from_pretrained(
    pretrained_retriever,

    # dataset to use for retrieval RAG, "wiki_dpr" by default
    #dataset="wiki_dpr",

    # use the dummy dataset
    use_dummy_dataset=True,

    # specifies number of documents to retrieve, 5 by default
    # not sure yet how this is used
    #n_docs=5,

    # used for FAISS index, needs to be "exact" as far as I understand
    # not exactly sure what this implies
    index_name="exact",
)

# the model uses the retriever to filter relevant content found in the documents and uses this to output a useful answer
model = RagTokenForGeneration.from_pretrained(pretrained_model, retriever=retriever)

input_dict = tokenizer.prepare_seq2seq_batch("who holds the record in 100m freestyle", return_tensors="pt")

generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
