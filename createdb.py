#load the pdf
#split into chunks
#create the embeddings
#Store into chromadb

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader,TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
load_dotenv()

data = PyPDFLoader("Document_loaders/book.pdf")
loader = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks= splitter.split_documents(loader)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="chroma-db",
)