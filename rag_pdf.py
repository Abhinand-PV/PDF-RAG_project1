import argparse
import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file for local development
load_dotenv()

# Default configuration settings
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))

def validate_environment():
    """Verify that required environment variables are set."""
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is not set.")
        print("Please create a .env file with your GROQ_API_KEY (see .env.example).")
        sys.exit(1)

def load_and_chunk_pdf(pdf_path, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Load a PDF and split it into chunks."""
    try:
        # Extracting text from every page of the given PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from {pdf_path}")

        # Splitting the pages into smaller, overlapping chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_documents(pages)
        print(f"Split into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
        return pages, chunks
    except Exception as e:
        print(f"Error loading and chunking PDF '{pdf_path}': {e}")
        sys.exit(1)

def create_vector_store(chunks, db_dir=DEFAULT_CHROMA_DB_DIR, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """Embed chunks and store in ChromaDB."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_dir,
        )
        print(f"Vector store created and persisted to {db_dir}")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        sys.exit(1)

def ask_question_with_sources(vector_store, llm, question, top_k=DEFAULT_TOP_K):
    """Retrieve relevant chunks and generate an answer with citations."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            print("No relevant context found in the document.")
            return

        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            page_num = doc.metadata.get("page", "unknown")
            page_str = int(page_num) + 1 if isinstance(page_num, int) or str(page_num).isdigit() else page_num
            context_parts.append(f"[Source {i} - Page {page_str}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a helpful assistant that answers questions based on the "
                "provided context. Each piece of context is labeled with its source "
                "number and page. In your answer, cite which sources you used "
                "(e.g., [Source 1], [Source 2]). If the answer is not in the context, "
                "say 'I don't have enough information to answer that based on the "
                "document.' Keep answers concise.\n\n"
                "Context:\n{context}"
            )),
            ("human", "{question}"),
        ])

        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})

        print(f"\nAnswer: {response.content}")
        print("\n--- Sources ---")
        for i, doc in enumerate(relevant_docs, 1):
            page_num = doc.metadata.get("page", "unknown")
            page_display = int(page_num) + 1 if isinstance(page_num, int) or str(page_num).isdigit() else page_num
            snippet = doc.page_content[:150].replace("\n", " ")
            print(f"  [Source {i}] Page {page_display}: \"{snippet}...\"")
        print()
    except Exception as e:
        print(f"Error processing question: {e}")

def print_help():
    """Print available interactive commands."""
    print("\nAvailable Commands:")
    print("  help        - Show this help menu")
    print("  info        - Display current PDF and model configuration settings")
    print("  clear       - Clear the terminal screen")
    print("  quit / exit - Exit the interactive assistant\n")

def print_info(pdf_path, total_pages, total_chunks, model_name, top_k, db_dir):
    """Display active session metadata."""
    print("\n--- RAG Session Info ---")
    print(f"  PDF Document  : {pdf_path}")
    print(f"  Total Pages   : {total_pages}")
    print(f"  Total Chunks  : {total_chunks}")
    print(f"  LLM Model     : {model_name}")
    print(f"  Top-K Sources : {top_k}")
    print(f"  Vector DB Dir : {db_dir}\n")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Local-first PDF RAG CLI tool using ChromaDB and Groq Cloud LLMs.")
    parser.add_argument("-p", "--pdf", type=str, help="Path to PDF file")
    parser.add_argument("-c", "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Chunk size for splitting (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("-o", "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help=f"Chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Top-K source chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("-m", "--model", type=str, default="llama-3.3-70b-versatile", help="Groq LLM model name (default: llama-3.3-70b-versatile)")
    parser.add_argument("-e", "--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help=f"Embedding model name (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("-d", "--db-dir", type=str, default=DEFAULT_CHROMA_DB_DIR, help=f"Chroma DB persistence directory (default: {DEFAULT_CHROMA_DB_DIR})")
    return parser.parse_args()

def main():
    validate_environment()
    args = parse_args()

    pdf_path = args.pdf
    if not pdf_path:
        pdf_path = input("Enter the path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return

    print("\n--- Loading and chunking PDF ---")
    pages, chunks = load_and_chunk_pdf(pdf_path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    print("\n--- Creating vector store ---")
    vector_store = create_vector_store(chunks, db_dir=args.db_dir, embedding_model=args.embedding_model)

    print(f"\n--- Initializing LLM ({args.model} on Groq) ---")
    llm = ChatGroq(model=args.model, temperature=0.1)

    print("\n--- Ready! Ask questions about your PDF ---")
    print("Type 'help' for options or 'quit' to exit.\n")

    # Interactive Q&A loop
    while True:
        try:
            question = input("Your question: ").strip()
            if question.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if question.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            if question.lower() == "help":
                print_help()
                continue
            if question.lower() == "info":
                print_info(pdf_path, len(pages), len(chunks), args.model, args.top_k, args.db_dir)
                continue
            if not question:
                continue

            ask_question_with_sources(vector_store, llm, question, top_k=args.top_k)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
