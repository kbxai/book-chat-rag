from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)


loader = TextLoader("Document_loaders/test.txt")
docs = loader.load()

chunks = splitter.split_documents(docs)

print(chunks,len(chunks))
