import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# DeepEval
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel  

# CONFIG
PDF_PATH = "sample.pdf"
DB_DIR = "chroma_db"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# LOAD / CREATE DB 
def load_or_create_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(DB_DIR):
        print("Loading existing DB...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("Creating new DB from PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       
        chunk_overlap=150    

    docs = text_splitter.split_documents(documents)

    db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    db.persist()
    return db


# BUILD CHAIN (LCEL)
def build_chain(db):
    llm = ChatOllama(model=LLM_MODEL)

    retriever = db.as_retriever(
        search_kwargs={"k": 5}  
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the question using ONLY the context below.
If the question asks about capabilities, list all capabilities clearly.

Context:
{context}

Question:
{question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain, retriever


# CHAT LOOP
def chat():
    db = load_or_create_db()
    chain, retriever = build_chain(db)

    eval_model = OllamaModel(model="llama3")

    relevancy_metric = AnswerRelevancyMetric(model=eval_model)
    faithfulness_metric = FaithfulnessMetric(model=eval_model)

    print("\nRAG Chatbot Ready (type 'exit' to quit)\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        #Generate response
        response = chain.invoke(query)
        response_text = response.content

        
        docs = retriever.invoke(query)

        print("\nBot:", response_text)

        # DeepEval
        test_case = LLMTestCase(
            input=query,
            actual_output=response_text,
            retrieval_context=[doc.page_content for doc in docs]
        )

        print("\n--- Evaluation ---")
        print("Relevancy:", relevancy_metric.measure(test_case))
        print("Faithfulness:", faithfulness_metric.measure(test_case))
        print("-------------\n")

if __name__ == "__main__":
    chat()