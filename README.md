# 🧠 ClearView RAG: The Transparent Document AI

**ClearView** is an advanced Retrieval-Augmented Generation (RAG) application designed to solve the "Black Box" problem of AI. Unlike standard chatbots, ClearView visualizes its entire thought process—showing you exactly how it expands your queries, searches your documents, and reranks results for accuracy.

## ✨ Key Features

* **🔍 Query Expansion:** Automatically generates 5 variations of your question to catch synonyms and implied intent.
* **⚖️ Cross-Encoder Reranking:** Uses **FlashRank** to "grade" retrieved documents and filter out irrelevant noise before the LLM answers.
* **🕵️ Transparency Board:** A dedicated UI pane that reveals the "Brain" of the AI (Retrieval stats, Confidence scores).
* **💾 Persistent Memory:** Your uploaded documents are stored permanently in a local vector database (`chroma_db_storage`), so you don't need to re-upload them every time.
* **🔗 Deep Tracing:** Integrated with **LangSmith** for line-by-line debugging and execution traces.

---

## 📂 Project Structure

```text
.
├── .venv/                     # Virtual environment (managed by uv)
├── chroma_db_storage/         # Persistent Vector Database (Created automatically)
├── generate_pdfs/             # 🛠️ FIRST TIME SETUP TOOLS
│   ├── sample_pdfs/           # Generated PDF files land here
│   └── generate_pdfs.py       # Script to create dummy leadership documents
├── .env                       # API Keys & Config (Not committed to Git)
├── .gitignore                 # Security rules for Git
├── app.py                     # 🚀 MAIN APPLICATION
├── pyproject.toml             # Dependency configuration
└── uv.lock                    # Exact version lockfile
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python 3.12** installed.
* **uv** package manager installed (faster and cleaner than pip).
    * *To install uv:* `pip install uv`

### 2. Installation
Clone the repository and sync the environment dependencies:

```bash
uv sync
```

### 3. Configuration
Create a `.env` file in the root directory with your keys:

```ini
OPENAI_API_KEY=sk-your-openai-key-here
# Optional: For deep tracing in LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2-your-langsmith-key
```

---

## 🛠️ First Run: Generating Data
**You only need to do this once.**
Before running the app, you need documents to search. We have a dedicated process for generating high-quality sample data.

1.  Navigate to the generator folder (or run from root):
    ```bash
    python generate_pdfs/generate_pdfs.py
    ```
2.  This will create 10 professional PDF guides (Leadership, Agile, Conflict Resolution, etc.) inside the `generate_pdfs/sample_pdfs/` folder.

---

## 🖥️ Running the Application

Once your data is ready (or if you have your own PDFs), launch the main interface:

```bash
streamlit run app.py
```

### How to use the App:
1.  **Load Data:** Open the **Left Sidebar**. Drag and drop the PDFs from `generate_pdfs/sample_pdfs/` (or your own files).
2.  **Process:** Click **"Process & Add to DB"**. Wait for the "Database Loaded" success message.
3.  **Chat:** Ask a question in the center chat box.
    * *Try:* "How do I handle a team that hates new software?"
4.  **Inspect:** Look at the **Right Pane (Transparency Board)** to see:
    * How the AI rewrote your question.
    * How many documents it found.
    * Why it chose specific documents (FlashRank Scores).

---

## 🧠 Masterclass Logic Map: Under the Hood

This application uses a specific data flow architecture designed for transparency and robustness. Here is the line-by-line technical breakdown of how `app.py` functions.

### 1. The Setup (Configuration)
**Role:** Sets the stage before any app logic runs.
* **`load_dotenv()`:** The bouncer. It checks your `.env` file to ensure you have the "keys" (API tokens) to enter.
* **Imports:** Loads the toolboxes (`LangChain` for logic, `Streamlit` for UI, `Chroma` for memory).
* **`PERSIST_DIR`:** Defines the "Filing Cabinet" on your hard drive (`./chroma_db_storage`) where documents will live forever.
* **LangSmith Check:** A safety mechanism. It checks if the tracing tools are installed; if not, it creates a "fake" decorator so the app doesn't crash on machines without LangSmith.

### 2. The Librarian (Database Functions)
**Role:** Handling raw files. These functions only run when you upload a PDF or restart the app.
* **`get_embedding_function()`** *(Cached)*:
    * **Job:** Creates the "Translator" that turns text into numbers.
    * **Why Cached?** We don't want to re-hire the translator every time you click a button. We hire them once and keep them on retainer using `@st.cache_resource`.
* **`load_vectorstore()`**:
    * **Job:** Walks to the file cabinet (`PERSIST_DIR`). If it sees files, it opens the drawer so we can search. If empty, it returns `None`.
* **`add_documents_to_db(files)`**:
    * **Job:** The "ETL" (Extract, Transform, Load) worker.
    * **Sequence:**
        1.  **Extract:** Saves the uploaded bytes to a temp file so `PyPDFLoader` can read it.
        2.  **Transform:** `RecursiveCharacterTextSplitter` chops the PDF into 500-character index cards.
        3.  **Load:** `vectorstore.add_documents` sends those cards to the Embedding model (to get numbers) and files them in the cabinet.

### 3. The Brain (The RAG Pipeline)
**Role:** The Thinker. This function (`run_rag_pipeline`) contains the **entire intelligence** of your app.
* **The `@traceable` Decorator:** Wraps the entire function in a "Trace Bubble." Any step that happens inside this function is grouped together in LangSmith under one name: "ClearView RAG Pipeline".
* **Sequence of Events:**
    1.  **Query Expansion:** The LLM acts as a creative writer, generating 5 variations of your question.
    2.  **Retrieval:** The Retriever takes all 6 questions (1 original + 5 variations) and runs 6 searches against the database. It compiles a unique list of found documents.
    3.  **Reranking:** **FlashRank** acts as a strict editor. It reads the 20+ documents found and gives them a score (0-1). Low scores are tossed out.
    4.  **Generation:** The LLM acts as the final speaker. It reads the surviving documents and answers your question.

### 4. The Face (The UI Loop)
**Role:** The Interaction. Streamlit runs this section from top to bottom *every time* you interact.
1.  **Sidebar Logic:** Checks if the database exists and handles file uploads.
2.  **Session State Initialization:** Creates "Short-term Memory" (`st.session_state.messages`) so chat history doesn't vanish on refresh.
3.  **Chat Loop (The "Input/Process" Pattern):**
    * **Input Event:** `if prompt := st.chat_input...`: Takes text, saves to memory, and immediately refreshes (`st.rerun()`) so the user sees their message instantly.
    * **Processing Event:** `if ... last_message == "user"`: This block runs *after* the refresh. It triggers the pipeline, captures the `run_id` from the `collect_runs()` net, and refreshes again to show the answer.
4.  **Transparency Board (Right Pane):** Reads the `latest_trace` data from memory and visualizes the reranking scores and trace links.

### 5. The Tracing System (Run ID Lifecycle)
How does the "Transparency Board" get the exact link to the debug trace?
1.  **The Net (`collect_runs`):** We wrap the pipeline execution in a `with collect_runs() as cb:` block. This context manager captures any trace generated inside it.
2.  **The Capture:** `run_id = cb.traced_runs[0].id` extracts the unique UUID of the pipeline run.
3.  **The Handoff:** This `run_id` is saved to `st.session_state` and passed to the Right Pane to generate the deep link (`https://smith.langchain.com/.../r/{run_id}`).


---

## ❓ Troubleshooting

**OneDrive Error (Windows):**
If you see `os error 396` or `os error 32` regarding `.DS_Store` or hardlinks:
1.  Ensure you have a `uv.toml` file in the root directory.
2.  It should contain: `link-mode = "copy"`.
3.  Run `uv sync` again.