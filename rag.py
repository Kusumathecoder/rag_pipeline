from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS

loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="llama3.1")

query = "What is cloud?"

retrieved_docs = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n", response)

