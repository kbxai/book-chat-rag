from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Document_loaders/test.pdf")
pages = loader.load()

print(pages[0].page_content)