# Chat with your PDFs: A Local-first RAG CLI

I built this command-line tool to solve a simple problem: querying long, complex PDF documents locally, quickly, and securely without uploading them to third-party web apps or paying for OpenAI API credits. 

This project uses **LangChain** to coordinate the workflow, **ChromaDB** to store text embeddings locally, and **Groq Cloud** (running Llama 3.3 70B) for incredibly fast Q&A generation.

---

## Behind the Scenes (How it Works)

Here is a quick look at what happens under the hood when you feed the script a PDF:

```mermaid
flowchart TD
    subgraph Ingestion ["1. Processing the PDF (Offline)"]
        A[Your PDF Document] --> B[Extract text page-by-page]
        B --> C[Split into overlapping 1000-char chunks]
        C --> D[Embed chunks locally using sentence-transformers]
        D --> E[(Store in local ChromaDB folder)]
    end

    subgraph Query ["2. Asking a Question (Online)"]
        F[Your Question] --> G[Convert question to vector]
        G --> H[Retrieve top 3 matching chunks from ChromaDB]
        H --> I[Package matching text + page numbers]
        I --> J[Send prompt to Groq API]
        J --> K[Llama 3.3 70B synthesizes answer]
        K --> L[Print answer + precise source citations]
    end
```

---

## Features I Implemented

- **Smart Text Chunking:** Instead of splitting text blindly, I configured a `RecursiveCharacterTextSplitter` with 1000-character chunks and a 200-character overlap. This keeps key sentences intact and prevents context from being lost across chunk borders.
- **Zero-Cost Embeddings:** I used `sentence-transformers/all-MiniLM-L6-v2` to generate vector embeddings on your local machine. This download is small (~90MB), runs entirely on CPU, and means you don't need to pay for embedding API keys.
- **Persistent Local Database:** ChromaDB stores the vectors directly in a `./chroma_db` folder. This means once you ingest a PDF, you don't have to wait to chunk and embed it again on your next run.
- **Lightning-Fast Q&A:** I integrated **Groq's API** using `llama-3.3-70b-versatile` to handle reasoning. Groq's hardware yields blazing-fast response speeds.
- **No-Hallucination Citations:** The prompt template tells the LLM to ground its answers strictly in the retrieved text. If the answer isn't in the PDF, it says so. It also prints exactly which page numbers and snippets it used, making the answers fully auditable.

---

## How to Set it Up and Run It

### What you need
- Python 3.9 or higher installed on your system.
- A free Groq API Key. You can grab one in about 30 seconds at [console.groq.com](https://console.groq.com/).

### Installation & Setup

1. **Clone the project repository:**
   ```bash
   git clone https://github.com/your-username/PDF-RAG.git
   cd PDF-RAG
   ```

2. **Set up a clean virtual environment:**
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API key:**
   Duplicate the `.env.example` file and rename it to `.env`:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and paste your Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

---

## Running the Assistant

To start chatting with your document, run:
```bash
python rag_pdf.py
```
- It will prompt you for the path to your PDF file (e.g., `metadata.pdf`).
- The first time you run it, it will automatically download the embedding model (approx. 90MB).
- Type in your questions. You'll get answers back with sources.
- Type `quit` when you're done.

---

## Why did I choose these tools?

- **Why LangChain?** I wanted a framework that would allow me to swap components easily. If I want to switch from ChromaDB to Pinecone or from Groq to a local Ollama model in the future, it's just a few lines of code.
- **Why ChromaDB?** It runs entirely in-memory or locally, making it perfect for lightweight, serverless tools. There was no need to manage docker containers or spin up cloud databases.
- **Why Groq & Llama 3.3?** Standard cloud inference can feel sluggish. Groq is almost instantaneous, making terminal-based chat interfaces feel like a conversation rather than a compilation step. Additionally, I took advantage of Groq's generous free tier to get state-of-the-art model inference without incurring any costs.
