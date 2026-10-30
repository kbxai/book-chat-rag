import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. Setup & UI Configuration
# ==========================================
load_dotenv()
st.set_page_config(page_title="BookChat AI", page_icon="📚", layout="centered")

# ==========================================
# 2. Initialize Models (Cached for Performance)
# ==========================================
@st.cache_resource
def init_rag_components():
    # Load Embeddings & VectorDB
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory="./chroma-db", embedding_function=embedding_model)

    # Load LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    # Setup Prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the context provided. If the context does not contain the answer, say 'I don't know'."),
        ("human", "Context: {context}\n\nQuestion: {question}\nAnswer:")
    ])

    return embedding_model, vectorstore, llm, prompt_template

embedding_model, vectorstore, llm, prompt_template = init_rag_components()

# ==========================================
# 3. Sidebar: Book Upload & Processing
# ==========================================
with st.sidebar:
    st.header("📚 Knowledge Base Setup")
    st.write("Upload a book or document to chat with it.")

    uploaded_file = st.file_uploader("Upload PDF or TXT", type=['pdf', 'txt'])

    if st.button("Process Document", type="primary") and uploaded_file:
        with st.spinner("Reading & indexing document..."):
            # Save uploaded file to a temporary file (Loaders require a file path)
            file_extension = uploaded_file.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Load Document
                if file_extension == 'pdf':
                    loader = PyPDFLoader(tmp_file_path)
                else:
                    loader = TextLoader(tmp_file_path, encoding="utf-8")
                docs = loader.load()

                # Split Document into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # Add to Chroma Vectorstore
                vectorstore.add_documents(splits)
                st.success(f"Successfully indexed {len(splits)} chunks!")

            finally:
                # Clean up the temporary file
                os.unlink(tmp_file_path)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 4. Main Chat Interface
# ==========================================
st.title("📚 BookChat AI")
st.caption("Ask specific questions about the documents stored in your knowledge base.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a book in the sidebar, or ask me a question about documents already in the database."}]

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if query := st.chat_input("Ask a question about your book..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching book..."):
            # Setup Retriever using your exact settings
            retriever = vectorstore.as_retriever(
                search_type='mmr',
                search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
            )

            # Retrieve and format context
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Generate Answer
            final_prompt = prompt_template.format_prompt(context=context, question=query)
            response = llm.invoke(final_prompt.to_messages())
            answer = response.content

            st.markdown(answer)

    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
