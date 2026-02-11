from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set your data directory and FAISS database path
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

# Function to create vector database
def create_vector_db():
    # Load all PDF files from the data directory
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents from PDF files.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # Initialize HuggingFace sentence embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Create FAISS vector store from document chunks
    db = FAISS.from_documents(texts, embeddings)

    # Save FAISS vector store locally
    db.save_local(DB_FAISS_PATH)
    print(f"Vector database created and saved to {DB_FAISS_PATH}")

# Run the function if this file is executed directly
if __name__ == "__main__":
    create_vector_db()
