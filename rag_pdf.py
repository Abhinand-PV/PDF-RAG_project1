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

def load_and_chunk_pdf(pdf_path):
    """Load a PDF and split it into chunks."""
    # Extracting text from every page of the given PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {pdf_path}")

    # Spliting the pages into smaller, overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Embbed chunks and store in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    print("Vector store created and persisted to ./chroma_db")
    return vector_store

def ask_question_with_sources(vector_store, llm, question):
    """Retrieve relevant chunks and generate an answer with citations."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)

    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        page_num = doc.metadata.get("page", "unknown")
        context_parts.append(f"[Source {i} - Page {int(page_num) + 1}]\n{doc.page_content}")

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
        snippet = doc.page_content[:150].replace("\n", " ")
        print(f"  [Source {i}] Page {int(page_num) + 1}: \"{snippet}...\"")
    print()
    
def main():
    # Ask the user which PDF to load, or take from command line
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1].strip()
    else:
        pdf_path = input("Enter the path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return

    print("\n--- Loading and chunking PDF ---")
    chunks = load_and_chunk_pdf(pdf_path)

    print("\n--- Creating vector store (first run downloads embedding model ~90MB) ---")
    vector_store = create_vector_store(chunks)

    print("\n--- Initializing LLM (Llama 3.3 70B on Groq) ---")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

    print("\n--- Ready! Ask questions about your PDF ---")
    print("Type 'quit' to exit.\n")

    # Interactive Q&A loop
    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            print("Goodbye!")
            break
        if question.lower() == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        if not question:
            continue

        ask_question_with_sources(vector_store, llm, question)
    


if __name__ == "__main__":
    main()