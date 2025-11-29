import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY) from the .env file
load_dotenv()

# --- IMPORTS ---
# 1. Document Loaders: To read PDF files
from langchain_community.document_loaders import PyPDFLoader
# 2. Vector Stores: To store text as numbers (embeddings) for search
from langchain_community.vectorstores import Chroma
# 3. Rerankers: To grade/filter documents after retrieval (The "Judge")
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
# 4. Embeddings & LLMs: The "Brain" and "Translator" (Text <-> Numbers)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# 5. Text Splitters: To chop large PDFs into small, digestible chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 6. Retrievers: The actual search logic
from langchain.retrievers import ContextualCompressionRetriever
# 7. Core components: Prompts, parsers, and tracing
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tracers.context import collect_runs

# --- LANGSMITH SETUP ---
# This block attempts to import LangSmith for debugging traces.
# We wrap it in a try/except so the app doesn't crash if you haven't installed it.
try:
    from langsmith import Client, traceable
    has_langsmith = True
except ImportError:
    has_langsmith = False
    # If LangSmith isn't found, we create a dummy decorator so the code doesn't break
    def traceable(*args, **kwargs):
        def decorator(f): return f
        return decorator

# --- CONFIGURATION ---
st.set_page_config(
    page_title="ClearView: Persistent RAG",
    page_icon="🧠",
    layout="wide"
)

# This is where your vector database will be saved on your hard drive
PERSIST_DIR = "./chroma_db_storage"

# --- UI STYLING ---
# Simple CSS to hide the default Streamlit menu and clean up the look
st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stStatus {background-color: #f0f2f6; border-radius: 10px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---

@st.cache_resource
def get_embedding_function():
    """
    Returns the OpenAI Embedding model.
    Cached because we don't want to reload the model object on every rerun.
    """
    return OpenAIEmbeddings()

def load_vectorstore():
    """
    Checks if a database exists on disk.
    If yes, it loads it into memory so we can search it.
    """
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding_function())
    return None

def add_documents_to_db(uploaded_files):
    """
    The ETL (Extract, Transform, Load) Pipeline.
    1. Extract: Read text from PDFs.
    2. Transform: Split text into small chunks.
    3. Load: Embed text and save to ChromaDB.
    """
    if not uploaded_files:
        return None

    docs = []
    progress_bar = st.progress(0, text="Reading files...")
    
    # 1. Extract Phase
    for i, file in enumerate(uploaded_files):
        # We must save the uploaded bytes to a temp file because PyPDFLoader expects a real file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())
        os.remove(tmp_path) # Clean up temp file
        progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processed {file.name}")
    
    # 2. Transform Phase
    # We split by 500 chars with 100 overlap to ensure context isn't lost at the cut points
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # 3. Load Phase
    # This automatically converts text -> numbers and saves to the PERSIST_DIR
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR, 
        embedding_function=get_embedding_function()
    )
    vectorstore.add_documents(splits)
    progress_bar.empty()
    return vectorstore

def reset_database():
    """Nuclear option: Deletes the database folder from the disk."""
    import shutil
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    st.cache_resource.clear()
    st.rerun()

# --- HELPER: LINE LIST PARSER ---
# This little tool takes a string like "1. Question A\n2. Question B" 
# and turns it into a Python list ["Question A", "Question B"]
class LineListOutputParser(BaseOutputParser[list[str]]):
    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

# --- THE MASTER PIPELINE (Traceable) ---
# This is the "Brain" of the operation.
# The @traceable decorator groups everything inside this function into ONE tree in LangSmith.
@traceable(run_type="chain", name="ClearView RAG Pipeline")
def run_rag_pipeline(question, vectorstore):
    
    # 1. Setup Tools
    llm = ChatOpenAI(temperature=0)
    # We ask for 5 documents initially (k=5)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # --- PHASE 1: Query Expansion ---
    # Instead of searching for exactly what the user typed, we ask the AI to brainstorm
    # 5 different ways to ask the same question. This helps catch synonyms.
    query_gen_prompt = ChatPromptTemplate.from_template(
        """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )
    generate_queries_chain = query_gen_prompt | llm | LineListOutputParser()
    expanded_queries = generate_queries_chain.invoke({"question": question})
    
    # --- PHASE 2: Retrieval ---
    # We search the database for the original question PLUS the 5 new variations.
    # If the user asked "lunch money", the AI might search for "meal reimbursement".
    initial_docs = []
    seen_content = set() # To avoid duplicates
    all_queries = [question] + expanded_queries
    
    for q in all_queries:
        docs = base_retriever.invoke(q)
        for d in docs:
            # Only add the document if we haven't seen this exact text before
            if d.page_content not in seen_content:
                initial_docs.append(d)
                seen_content.add(d.page_content)
                
    # --- PHASE 3: Reranking (The Filter) ---
    # We might have found 20+ documents from all those queries. Some might be irrelevant.
    # FlashRank (Cross-Encoder) reads every document and scores it against the ORIGINAL question.
    compressor = FlashrankRerank()
    reranked_docs = compressor.compress_documents(documents=initial_docs, query=question)
    
    # --- PHASE 4: Generation ---
    # We stick the top surviving documents into the final prompt context.
    context_text = "\n\n".join([d.page_content for d in reranked_docs])
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": question})
    
    # We return all the intermediate data so the UI can show the "Transparency Board"
    return response, expanded_queries, initial_docs, reranked_docs

# --- LAYOUT SETUP ---

with st.sidebar:
    st.header("🗂️ Knowledge Base")
    
    if os.path.exists(PERSIST_DIR):
        st.success(f"✅ Database Loaded")
    else:
        st.info("No database found. Upload files to create one.")

    new_files = st.file_uploader("Add New Documents", type=["pdf"], accept_multiple_files=True)
    
    if new_files and st.button("📥 Process & Add to DB"):
        with st.spinner("Adding documents to persistent storage..."):
            add_documents_to_db(new_files)
            st.success("Documents added!")
            time.sleep(1)
            st.rerun()

    st.divider()
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        st.caption("🟢 LangSmith Tracing Active")
    
    if st.button("🗑️ Clear Database", type="primary"):
        reset_database()

# --- MAIN PAGE ---
st.title("🧠 ClearView: Transparent Document AI")

# Split screen: 65% Chat, 35% Insights
col1, col2 = st.columns([0.65, 0.35], gap="large")

# Session state ensures data persists when you click buttons
if "latest_trace" not in st.session_state:
    st.session_state.latest_trace = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CENTER PANE (Chat) ---
with col1:
    st.subheader("💬 Chat")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        
        vectorstore = load_vectorstore()
        if not vectorstore:
            st.error("Please upload documents in the sidebar first!")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("🧠 Processing Pipeline...", expanded=True)
            
            try:
                # collect_runs() captures the Trace ID of whatever happens inside the block
                with collect_runs() as cb:
                    
                    # Update status bar as we go
                    status.write("🔍 **Step 1: Expanding Query...**")
                    status.write("📂 **Step 2: Retrieving Documents...**")
                    status.write("⚖️ **Step 3: Reranking Results...**")
                    status.write("✍️ **Step 4: Generating Answer...**")
                    
                    # Trigger the Master Pipeline
                    response, expanded_queries, initial_docs, reranked_docs = run_rag_pipeline(prompt, vectorstore)
                    
                    status.update(label="✅ Answer Ready", state="complete", expanded=False)
                    st.markdown(response)
                    
                    # Grab the ID of the main run_rag_pipeline call
                    run_id = cb.traced_runs[0].id

                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Save all the interesting stats to session state so the Right Pane can read them
                st.session_state.latest_trace = {
                    "run_id": str(run_id),
                    "original_query": prompt,
                    "expanded_queries": expanded_queries,
                    "retrieved_count": len(initial_docs),
                    "reranked_docs": reranked_docs
                }
                st.rerun() # Refresh the screen to update the Right Pane
                
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {str(e)}")

# --- RIGHT PANE (Insights) ---
with col2:
    st.subheader("🕵️ Transparency Board")
    
    if st.session_state.latest_trace:
        trace = st.session_state.latest_trace
        
        # Default Link (Dashboard)
        ls_url = "https://smith.langchain.com/"
        
        # Try to get the specific "Deep Link" to the exact run tree
        if has_langsmith and trace['run_id']:
            try:
                client = Client()
                time.sleep(0.5) # Wait for server sync
                run = client.read_run(trace['run_id'])
                ls_url = run.url
            except Exception as e:
                pass

        # SECTION 1: LangSmith Trace Button
        with st.expander("🔗 **LangSmith Trace**", expanded=True):
            st.success(f"Run ID: `{trace['run_id']}`")
            st.markdown(f"[➡️ **Open Full Pipeline Trace**]({ls_url})", unsafe_allow_html=True)

        # SECTION 2: Visualizing Query Expansion
        with st.expander("🔍 **Query Expansion**", expanded=True):
            st.markdown(f"**Original:** *{trace['original_query']}*")
            st.markdown("**AI Variations:**")
            for q in trace['expanded_queries']:
                st.markdown(f"- *{q}*")

        # SECTION 3: Retrieval Stats
        with st.expander("📊 **Retrieval Stats**", expanded=True):
            st.metric("Total Docs Found", trace['retrieved_count'])
            st.metric("Docs Kept (Reranked)", len(trace['reranked_docs']))

        # SECTION 4: Reranking Scores (The "Why")
        with st.expander("⚖️ **FlashRank Scores**", expanded=True):
            for i, doc in enumerate(trace['reranked_docs']):
                score = doc.metadata.get('relevance_score', 0)
                # Draw a visual green bar for relevance
                bar = "🟩" * int(score * 10) + "⬜" * (10 - int(score * 10))
                st.markdown(f"**Rank {i+1}** (Score: `{score:.3f}`)  \n{bar}  \n*{doc.page_content[:80]}...*")
                st.divider()
    else:
        st.info("Ask a question to see the internal reasoning process here.")