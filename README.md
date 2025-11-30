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

## 🧠 How it Works (The Pipeline)

1.  **User Question:** "My team is disconnected."
2.  **Expansion:** LLM generates: *"Strategies for remote team bonding", "Improving hybrid culture", etc.*
3.  **Retrieval:** The system searches the **ChromaDB** for *all* these variations, casting a wide net.
4.  **Reranking (The Filter):** **FlashRank** reads the retrieved text and scores it (0.0 to 1.0) based on relevance to your original specific problem. Low-score documents are discarded.
5.  **Generation:** The top documents are sent to GPT-4o to generate the final answer.

## ❓ Troubleshooting

**OneDrive Error (Windows):**
If you see `os error 396` or `os error 32` regarding `.DS_Store` or hardlinks:
1.  Ensure you have a `uv.toml` file in the root directory.
2.  It should contain: `link-mode = "copy"`.
3.  Run `uv sync` again.