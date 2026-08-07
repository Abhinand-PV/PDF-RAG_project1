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

## Live Demo

Here is the tool running in the terminal:

### 1. Document Ingestion & Local Setup
On the first run, the script loads the PDF, splits it into semantic chunks, downloads the embedding model, and builds the local Chroma database:

![Document Ingestion](pics/ingestion_demo.png)

### 2. Conversational Q&A with Page Citations
Ask any question, and the assistant responds with the answer and lists the exact page numbers and snippets used for the citation. If the answer is not in the text, it avoids hallucinating:

![Q&A Session](pics/qa_demo.png)

### 3. Programmatic Verification
You can also run independent tests to inspect metadata elements (like chunk sizes, total pages, and creator tags) extracted from the PDF:

![Programmatic Verification](pics/verification_demo.png)

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

To start chatting with your document using interactive prompt mode:
```bash
python rag_pdf.py
```

### CLI Command Options

You can also customize execution flags directly from the command line:

```bash
# Provide PDF path directly
python rag_pdf.py -p metadata.pdf

# Retrieve top 5 matching sources with custom chunk size
python rag_pdf.py -p metadata.pdf -k 5 -c 1200 -o 250

# Use a different Groq LLM model or Chroma DB folder
python rag_pdf.py -p metadata.pdf -m llama-3.3-70b-versatile -d ./my_chroma_db
```

| Flag | Long Option | Description | Default |
|------|-------------|-------------|---------|
| `-p` | `--pdf` | Path to input PDF document | Interactive Prompt |
| `-k` | `--top-k` | Number of context chunks retrieved | `3` |
| `-c` | `--chunk-size` | Character size per split chunk | `1000` |
| `-o` | `--chunk-overlap` | Overlap characters between chunks | `200` |
| `-m` | `--model` | Groq LLM model identifier | `llama-3.3-70b-versatile` |
| `-e` | `--embedding-model` | HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `-d` | `--db-dir` | Local directory for Chroma DB storage | `./chroma_db` |

### Interactive Commands

During an active session, you can enter the following utility commands into the prompt:
- `help` — View available commands
- `info` — View document metadata, active model name, top-k retrieval setting, and chunk counts
- `clear` — Clear the terminal screen
- `quit` / `exit` — Exit the application

---


## Why did I choose these tools?

- **Why LangChain?** I wanted a framework that would allow me to swap components easily. If I want to switch from ChromaDB to Pinecone or from Groq to a local Ollama model in the future, it's just a few lines of code.
- **Why ChromaDB?** It runs entirely in-memory or locally, making it perfect for lightweight, serverless tools. There was no need to manage docker containers or spin up cloud databases.
- **Why Groq & Llama 3.3?** Standard cloud inference can feel sluggish. Groq is almost instantaneous, making terminal-based chat interfaces feel like a conversation rather than a compilation step. Additionally, I took advantage of Groq's generous free tier to get state-of-the-art model inference without incurring any costs.

---

## How the Project Works

In summary, this RAG application operates through a two-phase pipeline:

1. **Document Ingestion & Indexing:** 
   - The user provides a path to a PDF document.
   - The system extracts the raw text page-by-page and segments it into overlapping chunks to preserve contextual boundaries.
   - These chunks are converted into dense vector representations locally using a lightweight HuggingFace transformer model.
   - The vectors and their corresponding text segments are indexed and stored in a local, persistent vector database (ChromaDB).

2. **Retrieval & Answer Synthesis:**
   - When a user submits a query, it is converted into a vector representation using the same embedding model.
   - The vector database performs a semantic similarity search to retrieve the most relevant text chunks.
   - These chunks, alongside their page numbers, are compiled into a structured prompt context.
   - The context and user query are sent to the Groq API, where a Llama 3.3 model generates a precise, cited answer grounded strictly in the provided document text.
