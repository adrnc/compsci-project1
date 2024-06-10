import os
os.environ["USER_AGENT"] = "local"

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

model = "qwen:0.5b"
llm = Ollama(model=model)

# start ollama (https://github.com/ollama/ollama) locally like:
# ollama run "model"

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits,
    #persist_directory='chroma',
    embedding=OllamaEmbeddings(model=model, show_progress=True),
)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is Task Decomposition?")

print(answer)
'''
Task Decomposition can be done using human inputs. For example, they could ask questions such as:

1. What is the main purpose of the task?
2. How would you improve the efficiency of the task?
3. How would you ensure that the task does not rely on a long context window to communicate over complicated task content?
'''
