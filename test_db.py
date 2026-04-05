from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

DB_DIR = "chroma_db"

embeddings = OllamaEmbeddings(model="llama3")

db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

# Test query
docs = db.similarity_search("What is GenAI?", k=3)

print("\n Retrieved Chunks:\n")
for doc in docs:
    print(doc.page_content)
    print("-" * 50)