import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv

# Load environment variables explicitly
load_dotenv()

# --- IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tracers.context import collect_runs

# Try to import LangSmith Client for deep linking (fails gracefully if not set up)
try:
    from langsmith import Client
    has_langsmith = True
except ImportError:
    has_langsmith = False

# --- CONFIGURATION ---
st.set_page_config(
    page_title="ClearView: Persistent RAG",
    page_icon="🧠",
    layout="wide"
)

# Persistent Directory for ChromaDB
PERSIST_DIR = "./chroma_db_storage"

# --- UI STYLING ---
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
    return OpenAIEmbeddings()

def load_vectorstore():
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding_function())
    return None

def add_documents_to_db(uploaded_files):
    if not uploaded_files:
        return None

    docs = []
    progress_bar = st.progress(0, text="Reading files...")
    
    for i, file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())
        os.remove(tmp_path)
        progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processed {file.name}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR, 
        embedding_function=get_embedding_function()
    )
    vectorstore.add_documents(splits)
    progress_bar.empty()
    return vectorstore

def reset_database():
    import shutil
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    st.cache_resource.clear()
    st.rerun()

# --- HELPER: LINE LIST PARSER ---
class LineListOutputParser(BaseOutputParser[list[str]]):
    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

# --- THE MASTER PIPELINE (Traceable) ---
def run_rag_pipeline(question, vectorstore):
    
    # 1. Setup Tools
    llm = ChatOpenAI(temperature=0)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # --- PHASE 1: Query Expansion ---
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
    initial_docs = []
    seen_content = set() 
    all_queries = [question] + expanded_queries
    
    for q in all_queries:
        docs = base_retriever.invoke(q)
        for d in docs:
            if d.page_content not in seen_content:
                initial_docs.append(d)
                seen_content.add(d.page_content)
                
    # --- PHASE 3: Reranking ---
    compressor = FlashrankRerank()
    reranked_docs = compressor.compress_documents(documents=initial_docs, query=question)
    
    # --- PHASE 4: Generation ---
    context_text = "\n\n".join([d.page_content for d in reranked_docs])
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": question})
    
    return response, expanded_queries, initial_docs, reranked_docs

# --- LAYOUT SETUP ---

# 1. LEFT PANE (Sidebar)
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

col1, col2 = st.columns([0.65, 0.35], gap="large")

# Initialize Session State
if "latest_trace" not in st.session_state:
    st.session_state.latest_trace = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CENTER PANE (Chat) ---
with col1:
    st.subheader("💬 Chat")
    
    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Logic
    # We check if the last message was NOT from the assistant to trigger a response
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() # Force a rerun to display the user message immediately

    # If the last message is from the user, generate the AI response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        
        user_question = st.session_state.messages[-1]["content"]
        
        vectorstore = load_vectorstore()
        if not vectorstore:
            st.error("Please upload documents in the sidebar first!")
        else:
            with st.chat_message("assistant"):
                status = st.status("🧠 Processing Pipeline...", expanded=True)
                
                try:
                    # Capture Trace
                    with collect_runs() as cb:
                        
                        status.write("🔍 **Step 1: Expanding Query...**")
                        status.write("📂 **Step 2: Retrieving Documents...**")
                        status.write("⚖️ **Step 3: Reranking Results...**")
                        status.write("✍️ **Step 4: Generating Answer...**")
                        
                        response, expanded_queries, initial_docs, reranked_docs = run_rag_pipeline(user_question, vectorstore)
                        
                        status.update(label="✅ Answer Ready", state="complete", expanded=False)
                        st.markdown(response)
                        
                        run_id = cb.traced_runs[0].id

                    # Append AI response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Update Right Pane
                    st.session_state.latest_trace = {
                        "run_id": str(run_id),
                        "original_query": user_question,
                        "expanded_queries": expanded_queries,
                        "retrieved_count": len(initial_docs),
                        "reranked_docs": reranked_docs
                    }
                    st.rerun() # Rerun one last time to update the Right Pane
                    
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error: {str(e)}")

# --- RIGHT PANE (Insights) ---
with col2:
    st.subheader("🕵️ Transparency Board")
    
    if st.session_state.latest_trace:
        trace = st.session_state.latest_trace
        
        ls_url = "https://smith.langchain.com/"
        
        if has_langsmith and trace['run_id']:
            try:
                client = Client()
                time.sleep(0.5)
                run = client.read_run(trace['run_id'])
                ls_url = run.url
            except Exception as e:
                pass

        with st.expander("🔗 **LangSmith Trace**", expanded=True):
            st.success(f"Run ID: `{trace['run_id']}`")
            st.markdown(f"[➡️ **Open Full Pipeline Trace**]({ls_url})", unsafe_allow_html=True)

        with st.expander("🔍 **Query Expansion**", expanded=True):
            st.markdown(f"**Original:** *{trace['original_query']}*")
            st.markdown("**AI Variations:**")
            for q in trace['expanded_queries']:
                st.markdown(f"- *{q}*")

        with st.expander("📊 **Retrieval Stats**", expanded=True):
            st.metric("Total Docs Found", trace['retrieved_count'])
            st.metric("Docs Kept (Reranked)", len(trace['reranked_docs']))

        with st.expander("⚖️ **FlashRank Scores**", expanded=True):
            for i, doc in enumerate(trace['reranked_docs']):
                score = doc.metadata.get('relevance_score', 0)
                bar = "🟩" * int(score * 10) + "⬜" * (10 - int(score * 10))
                st.markdown(f"**Rank {i+1}** (Score: `{score:.3f}`)  \n{bar}  \n*{doc.page_content[:80]}...*")
                st.divider()
    else:
        st.info("Ask a question to see the internal reasoning process here.")