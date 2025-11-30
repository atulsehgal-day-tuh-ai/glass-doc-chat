import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------

# Load environment variables (OPENAI_API_KEY, LANGCHAIN_API_KEY) immediately.
# This must happen before we import LangChain classes so they can find the keys.
load_dotenv()

# --- LANGCHAIN IMPORTS ---
# PyPDFLoader: The specific tool that knows how to read binary PDF files.
from langchain_community.document_loaders import PyPDFLoader

# Chroma: Our Vector Database. It stores "Embeddings" (number lists) not just text.
from langchain_community.vectorstores import Chroma

# FlashRank: The "Cross-Encoder". Unlike standard retrievers that look at general 
# similarity (cosine), this model reads the query and document side-by-side 
# to give a precise 0.0-1.0 relevance score. It runs locally on your CPU.
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# OpenAIEmbeddings: The "Translator" that turns text into vectors (arrays of numbers).
# ChatOpenAI: The "Brain" (GPT-3.5/4) that generates the final answer.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# RecursiveCharacterTextSplitter: The "Scissors". It smartly cuts documents into chunks.
# It tries to keep paragraphs together, then sentences, then words.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Retrievers & Chains: The building blocks of our logic.
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnablePassthrough

# collect_runs: The "Net" for LangSmith. 
# Even without a decorator, this Context Manager catches every trace ID generated inside it.
from langchain_core.tracers.context import collect_runs

# --- OPTIONAL IMPORTS ---
# We wrap this in try/except so your app doesn't crash if 'langsmith' isn't installed.
try:
    from langsmith import Client
    has_langsmith = True
except ImportError:
    has_langsmith = False

# Streamlit Page Config: Sets the browser tab title and layout width.
st.set_page_config(page_title="ClearView: Persistent RAG", page_icon="🧠", layout="wide")

# The folder where ChromaDB will save its files on your hard drive.
PERSIST_DIR = "./chroma_db_storage"

# Custom CSS: Hides the "Deploy" button and cleans up the UI.
st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stStatus {background-color: #f0f2f6; border-radius: 10px; padding: 10px;}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. BACKEND FUNCTIONS (Cached for Performance)
# -----------------------------------------------------------------------------

# @st.cache_resource is CRITICAL. 
# Streamlit re-runs the entire script every time you click a button.
# Without caching, we would reconnect to OpenAI and reload the DB 50 times a second.
@st.cache_resource
def get_embedding_function():
    return OpenAIEmbeddings()

def load_vectorstore():
    """Load the existing DB from disk if it exists."""
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding_function())
    return None

def add_documents_to_db(uploaded_files):
    """
    The ETL Pipeline (Extract, Transform, Load).
    This runs only when you upload new files.
    """
    if not uploaded_files:
        return None

    docs = []
    progress_bar = st.progress(0, text="Reading files...")
    
    for i, file in enumerate(uploaded_files):
        # STREAMLIT QUIRK: Uploaded files are bytes in RAM. 
        # PyPDFLoader needs a physical file path.
        # So we create a temporary file, write the bytes to it, read it, then delete it.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())
        os.remove(tmp_path) # Cleanup
        progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processed {file.name}")
    
    # SPLITTING: We use 500 chars with 100 char overlap.
    # Overlap is crucial so words at the cut point don't lose their context.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # INDEXING: This sends text to OpenAI to get numbers, then saves to disk.
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR, 
        embedding_function=get_embedding_function()
    )
    vectorstore.add_documents(splits)
    progress_bar.empty()
    return vectorstore

def reset_database():
    """Nuclear option: Deletes the folder to clear memory."""
    import shutil
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    st.cache_resource.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# 3. HELPER CLASS FOR QUERY EXPANSION
# -----------------------------------------------------------------------------
class LineListOutputParser(BaseOutputParser[list[str]]):
    """
    The LLM outputs text like:
    1. Question A
    2. Question B
    
    This parser cleans it up into a Python list: ["Question A", "Question B"]
    """
    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]


# -----------------------------------------------------------------------------
# 4. THE MASTER PIPELINE (The "Brain")
# -----------------------------------------------------------------------------
# This function is not decorated with @traceable because we use 'collect_runs()'
# in the main loop, which captures everything inside this function automatically.
# Also, the settings for tracing are set via ENV variables in the .env file.

def run_rag_pipeline(question, vectorstore):
    
    # 1. SETUP
    llm = ChatOpenAI(temperature=0)
    # k=5 means "Get the top 5 matches" from the database initially.
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 2. QUERY EXPANSION (The "Creative" Step)
    # We ask GPT to brainstorm synonyms and variations.
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
    
    # 3. RETRIEVAL (The "Wide Net" Step)
    # We search the database 6 times (1 original + 5 variations).
    # We use a set() to ensure we don't end up with duplicate documents.
    initial_docs = []
    seen_content = set() 
    all_queries = [question] + expanded_queries
    
    for q in all_queries:
        docs = base_retriever.invoke(q)
        for d in docs:
            if d.page_content not in seen_content:
                initial_docs.append(d)
                seen_content.add(d.page_content)
                
    # 4. RERANKING (The "Filter" Step)
    # FlashRank looks at the 20+ docs we found and scores them.
    # It throws away low-score docs, leaving us with the "True" best matches.
    compressor = FlashrankRerank()
    reranked_docs = compressor.compress_documents(documents=initial_docs, query=question)
    
    # 5. GENERATION (The "Synthesis" Step)
    # We paste the text from the top docs into the prompt.
    context_text = "\n\n".join([d.page_content for d in reranked_docs])
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": question})
    
    # Return everything so the UI can show the transparency board
    return response, expanded_queries, initial_docs, reranked_docs


# -----------------------------------------------------------------------------
# 5. UI LAYOUT & INTERACTION LOOP
# -----------------------------------------------------------------------------

# LEFT SIDEBAR: File Management
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
            st.rerun() # Refresh the app to verify the DB exists now

    st.divider()
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        st.caption("🟢 LangSmith Tracing Active")
    
    if st.button("🗑️ Clear Database", type="primary"):
        reset_database()

# MAIN AREA: Split into Chat (Left) and Insights (Right)
st.title("🧠 ClearView: Transparent Document AI")
col1, col2 = st.columns([0.65, 0.35], gap="large")

# Initialize Session State
# This acts as the "Memory" of the app. It survives re-runs.
if "latest_trace" not in st.session_state:
    st.session_state.latest_trace = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CENTER PANE: CHAT INTERFACE ---
with col1:
    st.subheader("💬 Chat")
    
    # Render all previous messages from memory
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # CAPTURE USER INPUT
    # st.chat_input stops the script until the user hits Enter.
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() # Immediate rerun ensures the user's message appears instantly

    # PROCESS RESPONSE (Only if the last message was a user)
    # This logic runs immediately AFTER the st.rerun() above.
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        
        user_question = st.session_state.messages[-1]["content"]
        
        vectorstore = load_vectorstore()
        if not vectorstore:
            st.error("Please upload documents in the sidebar first!")
        else:
            with st.chat_message("assistant"):
                status = st.status("🧠 Processing Pipeline...", expanded=True)
                
                try:
                    # TRACING BLOCK:
                    # 'collect_runs()' is a context manager. It watches everything that happens inside it.
                    # It grabs the Trace ID of the 'run_rag_pipeline' call.
                    with collect_runs() as cb:
                        
                        # UI Feedback
                        status.write("🔍 **Step 1: Expanding Query...**")
                        status.write("📂 **Step 2: Retrieving Documents...**")
                        status.write("⚖️ **Step 3: Reranking Results...**")
                        status.write("✍️ **Step 4: Generating Answer...**")
                        
                        # EXECUTE LOGIC
                        response, expanded_queries, initial_docs, reranked_docs = run_rag_pipeline(user_question, vectorstore)
                        
                        status.update(label="✅ Answer Ready", state="complete", expanded=False)
                        st.markdown(response)
                        
                        # Capture the specific ID of this run for the Deep Link
                        run_id = cb.traced_runs[0].id

                    # Save to memory
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Update the "Transparency Board" data in memory
                    st.session_state.latest_trace = {
                        "run_id": str(run_id),
                        "original_query": user_question,
                        "expanded_queries": expanded_queries,
                        "retrieved_count": len(initial_docs),
                        "reranked_docs": reranked_docs
                    }
                    st.rerun() # Rerun again to force the Right Pane to update with new data
                    
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error: {str(e)}")

# --- RIGHT PANE: INSIGHTS BOARD ---
# This is the "Transparency Board" that shows the AI's "thought process".
# It only renders if we have trace data in the session state.
with col2:
    st.subheader("🕵️ Transparency Board")
    
    # Check if a trace exists (meaning a question has been asked)
    if st.session_state.latest_trace:
        trace = st.session_state.latest_trace
        
        # --- SECTION 1: LANGSMITH LINK ---
        # Construct the URL to the LangSmith dashboard.
        ls_url = "https://smith.langchain.com/"
        
        # Try to get the specific "Deep Link" to the exact run tree if LangSmith is enabled.
        if has_langsmith and trace['run_id']:
            try:
                client = Client()
                time.sleep(0.5) # Wait briefly for the trace to propagate to the LangSmith server
                run = client.read_run(trace['run_id'])
                ls_url = run.url
            except Exception as e:
                # If fetching the specific URL fails, we fall back to the main dashboard URL.
                pass

        # Use an expander to keep the UI clean, but default it to open (expanded=True)
        with st.expander("🔗 **LangSmith Trace**", expanded=True):
            st.success(f"Run ID: `{trace['run_id']}`")
            st.markdown(f"[➡️ **Open Full Pipeline Trace**]({ls_url})", unsafe_allow_html=True)

        # --- SECTION 2: QUERY EXPANSION VISUALIZATION ---
        # Shows the user how their original query was transformed into multiple search queries.
        with st.expander("🔍 **Query Expansion**", expanded=True):
            st.markdown(f"**Original:** *{trace['original_query']}*")
            st.markdown("**AI Variations:**")
            for q in trace['expanded_queries']:
                st.markdown(f"- *{q}*")

        # --- SECTION 3: RETRIEVAL STATISTICS ---
        # Displays the "funnel" effect: how many docs were found vs how many were kept.
        with st.expander("📊 **Retrieval Stats**", expanded=True):
            st.metric("Total Docs Found", trace['retrieved_count'])
            st.metric("Docs Kept (Reranked)", len(trace['reranked_docs']))

        # --- SECTION 4: RERANKING SCORES ---
        # This loops through the documents that survived the FlashRank filter.
        # It displays their relevance score and a visual progress bar.
        with st.expander("⚖️ **FlashRank Scores**", expanded=True):
            for i, doc in enumerate(trace['reranked_docs']):
                # Flashrank adds a 'relevance_score' metadata field to each document
                score = doc.metadata.get('relevance_score', 0)
                # Create a simple visual bar using emoji blocks based on the score (0-10 scale)
                bar = "🟩" * int(score * 10) + "⬜" * (10 - int(score * 10))
                # Display rank, numerical score, visual bar, and a snippet of the content
                st.markdown(f"**Rank {i+1}** (Score: `{score:.3f}`)  \n{bar}  \n*{doc.page_content[:80]}...*")
                st.divider()
    else:
        # Placeholder text shown when the app first loads
        st.info("Ask a question to see the internal reasoning process here.")