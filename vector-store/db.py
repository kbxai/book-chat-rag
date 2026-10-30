from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

docs = [
    Document(
        page_content=(
            "Python is a high-level, interpreted programming language known "
            "for its simplicity and readability."
        ),
        metadata={"source": "doc1"},
    ),
    Document(
        page_content=(
            "Java is a versatile, object-oriented programming language widely "
            "used for building enterprise-level applications."
        ),
        metadata={"source": "doc2"},
    ),
    Document(
        page_content=(
            "JavaScript is a dynamic, high-level programming language "
            "primarily used for web development."
        ),
        metadata={"source": "doc3"},
    ),
]

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="chroma-db",
)
