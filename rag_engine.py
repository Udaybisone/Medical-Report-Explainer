import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from chromadb.config import Settings

def setup_rag():
    docs = []
    for file in os.listdir("health_knowledge_base"):
        if file.endswith(".txt"):
            path = os.path.join("health_knowledge_base", file)
            loader = TextLoader(path)
            docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    settings = Settings(
        chroma_api_impl="local",
        persist_directory=None,  # Important: disables persistent disk storage
        anonymized_telemetry=False,
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        client_settings=settings
    )
    retriever = vectordb.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
