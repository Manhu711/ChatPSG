import streamlit as st
import pandas as pd
import PyPDF2
import os
from pathlib import Path
import pickle
from datetime import datetime
import re
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import tempfile
from sentence_transformers import SentenceTransformer
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Optional
from langchain.schema import LLMResult
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.retrievers import (
    BM25Retriever,
)
from langchain.retrievers import (
    ContextualCompressionRetriever,
)
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.document_transformers import (
    EmbeddingsRedundantFilter,
)

# Set page config
st.set_page_config(
    page_title="ChatPSG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("ChatPSG")

st.write("""
This application allows you to search for keywords across PSG PDF files and chat with an AI about the content.
Choose your mode in the sidebar.
""")

# Constants
PDF_FOLDER = "pdfs"  # Update with your actual path
INDEX_FILE = "pdf_index.pkl"  # Define a constant for the index file name

# API Key handling - get from secrets for all users
try:
    GOOGLE_API_KEY = st.secrets["api_keys"]["GOOGLE_API_KEY"]
except Exception as e:
    st.error("Google API Key not found in Streamlit secrets.")
    st.info("""
    This app requires a Google Gemini API key to be configured by the administrator.
    
    If you are the app administrator, please add your API key to Streamlit secrets.
    """)
    print(f"API Key error: {str(e)}")
    st.stop()  # Stop execution until API key is provided

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default model, will be updated if embedding_model.txt exists

# Try to load the embedding model name from the file
try:
    embedding_model_path = "processed_data/embedding_model.txt"
    if os.path.exists(embedding_model_path):
        with open(embedding_model_path, 'r') as f:
            saved_model = f.read().strip()
            if saved_model:
                EMBEDDING_MODEL = saved_model
                print(f"Using embedding model from file: {EMBEDDING_MODEL}")
except Exception as e:
    print(f"Error loading embedding model name: {e}")
    print(f"Using default embedding model: {EMBEDDING_MODEL}")

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'current_sources' not in st.session_state:
    st.session_state.current_sources = None
if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = None
if 'show_diagnostics' not in st.session_state:
    st.session_state.show_diagnostics = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def load_csv_data():
    """Load and process the CSV file with additional information"""
    try:
        df = pd.read_csv('combined_list_with_filenames.csv')
        # Create a dictionary mapping filenames to their details
        return df.set_index('psg filename').to_dict('index')
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return {}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        st.error(f"Error reading PDF {pdf_path}: {str(e)}")
        return ""

def save_index(index_data, filename="pdf_index.pkl"):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(index_data, f)
        return True
    except Exception as e:
        st.error(f"Error saving index: {str(e)}")
        return False

def load_index(filename=INDEX_FILE):
    """Load the PDF index file"""
    try:
        # Try different possible locations for the index file
        possible_paths = [
            filename,  # Try the current directory
            os.path.join(os.getcwd(), filename),  # Try absolute path in current directory
            os.path.join(".", filename),  # Try explicitly in current directory
        ]
        
        # Print debug info about file paths
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for index file {filename} in possible locations:")
        for path in possible_paths:
            exists = os.path.exists(path)
            print(f"  - {path}: {'EXISTS' if exists else 'NOT FOUND'}")
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading index from: {path}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
        
        # If we get here, the file wasn't found
        st.warning(f"Index file {filename} not found in any of the expected locations")
        return None
    except Exception as e:
        st.warning(f"Error loading index: {str(e)}")
        return None

def index_needs_rebuild(index_file="pdf_index.pkl", pdf_folder=PDF_FOLDER):
    if not os.path.exists(index_file):
        return True
    
    try:
        index_time = os.path.getmtime(index_file)
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            return True
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            if os.path.getmtime(pdf_path) > index_time:
                return True
        
        index_data = load_index(index_file)
        if index_data and len(index_data["files"]) != len(pdf_files):
            return True
            
        return False
    except Exception as e:
        st.warning(f"Error checking index: {str(e)}")
        return True

def index_pdfs(pdf_folder=PDF_FOLDER, force_reindex=False):
    """Index all PDFs in the folder with cache option"""
    index_file = "pdf_index.pkl"
    csv_data = load_csv_data()
    
    if not force_reindex and not index_needs_rebuild(index_file, pdf_folder):
        index_data = load_index(index_file)
        if index_data:
            return index_data["contents"], index_data["files"]
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.error(f"No PDF files found in {pdf_folder}")
        return {}, []
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    pdf_contents = {}
    
    for i, pdf_file in enumerate(pdf_files):
        progress = (i + 1) / len(pdf_files)
        progress_bar.progress(progress)
        status_text.text(f"Indexing {i+1}/{len(pdf_files)}: {pdf_file}")
        
        pdf_path = os.path.join(pdf_folder, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        if text:
            pdf_contents[pdf_file] = {
                'text': text,
                'csv_data': csv_data.get(pdf_file, {})
            }
    
    progress_bar.empty()
    status_text.empty()
    
    index_data = {
        "contents": pdf_contents,
        "files": pdf_files,
        "timestamp": datetime.now().isoformat()
    }
    success = save_index(index_data)
    
    if success:
        st.success(f"Successfully indexed {len(pdf_files)} PDF files")
    
    return pdf_contents, pdf_files

def search_pdfs(keywords, pdf_contents, case_sensitive=False):
    """Search PDFs for keywords and return matching files with CSV data"""
    results = []
    
    for pdf_file, data in pdf_contents.items():
        if not isinstance(data, dict) or 'text' not in data:
            continue
            
        text = data['text']
        csv_data = data.get('csv_data', {})
        
        all_found = True
        for keyword in keywords:
            if case_sensitive:
                if keyword not in text:
                    all_found = False
                    break
            else:
                if keyword.lower() not in text.lower():
                    all_found = False
                    break
        
        if all_found:
            results.append({
                'filename': pdf_file,
                'active_ingredient': csv_data.get('Active Ingredient (link to Specific Guidance)', 'N/A'),
                'url': csv_data.get('URL', 'N/A'),
                'route': csv_data.get('Route', 'N/A'),
                'dosage_form': csv_data.get('Dosage Form', 'N/A')
            })
    
    return results

def display_results(results, keywords):
    """Display search results in a user-friendly format"""
    if results:
        st.success(f"Found {len(results)} matching files containing all keywords: {', '.join(keywords)}")
        
        # Create a clean DataFrame with exactly the columns we want
        data = []
        for i, result in enumerate(results):
            data.append({
                'No.': i + 1,
                'Active Ingredient': result.get('active_ingredient', 'N/A'),
                'Route': result.get('route', 'N/A'),
                'Dosage Form': result.get('dosage_form', 'N/A'),
                'URL': result.get('url', 'N/A')
            })
        
        # Create DataFrame and reorder columns
        result_df = pd.DataFrame(data)
        
        # Use st.table which typically doesn't show the DataFrame index
        st.table(result_df)
        
        # Option to download results as CSV
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results as CSV",
            csv,
            "psg_search_results.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.warning("No matching files found for all keywords.")

def setup_vector_store():
    """Load the preprocessed vector store"""
    vector_store_path = "processed_data/vector_store.faiss"
    if not os.path.exists(vector_store_path):
        st.error("Vector store not found. Please run preprocess_docs.py first.")
        return None
    
    try:
        # Initialize embeddings if not already done
        if st.session_state.embeddings is None:
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Add allow_dangerous_deserialization=True since we trust our own processed data
        vector_store = FAISS.load_local(
            vector_store_path, 
            st.session_state.embeddings,
            allow_dangerous_deserialization=True  # Safe because we created the data ourselves
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def get_mmr_retriever(vector_store, num_chunks=10):
    """
    Create an MMR retriever for better diverse results
    """
    if vector_store is None:
        st.error("Vector store not available")
        return None
        
    # Maximum marginal relevance search
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": num_chunks,
            "fetch_k": num_chunks * 3,  # Fetch more docs, then rerank
            "lambda_mult": 0.7,  # Balance between relevance (1.0) and diversity (0.0)
        }
    )

class DebugCallbackHandler(BaseCallbackHandler):
    """Callback handler for debugging that captures prompts and responses."""
    def __init__(self):
        self.prompt = None
        self.response = None
        self.last_prompt = None
        self.last_response = None

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Capture the prompt when LLM starts."""
        self.last_prompt = prompts[0]
        self.prompt = self.last_prompt
        print(f"Debug - Captured prompt: {self.prompt}")
        # Store the prompt in session state
        st.session_state.last_prompt = self.prompt

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Capture the response when LLM ends."""
        try:
            if response.generations and len(response.generations) > 0:
                self.last_response = response.generations[0][0].text
                self.response = self.last_response
                print(f"Debug - Captured response: {self.response}")
        except Exception as e:
            print(f"Debug - Error capturing response: {str(e)}")
            self.response = str(response)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Handle LLM errors."""
        print(f"Debug - LLM Error: {str(error)}")

def setup_chat_chain():
    """Set up the chat chain with Gemini with chat history"""
    vector_store = setup_vector_store()
    if vector_store is None:
        return None

    # Create a debug callback handler
    debug_handler = DebugCallbackHandler()

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.5,
        max_output_tokens=8192,
        google_api_key=GOOGLE_API_KEY,
        callbacks=[debug_handler]  # Add callback handler
    )

    # Create a prompt template with chat history
    template = """You are a helpful assistant analyzing PSG (Product Specific Guidance) documents. 
    Answer the user's question based on the relevant information you find in the documents.
    Be thorough, accurate, and provide specific details from the documents.
    
    If the provided context doesn't contain information about the user's question, explicitly state that the PSG documents don't contain information on that topic.
    
    Chat History:
    {chat_history}
    
    Context:
    {context}

    Question: {question}

    Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    # Get number of chunks from session state
    num_chunks = st.session_state.get('num_chunks', 10)
    
    # Using only MMR search
    retriever = get_mmr_retriever(vector_store, num_chunks)
    
    if retriever is None:
        st.error("Failed to create retriever")
        return None

    # Initialize memory with conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Changed from "result" to "answer"
    )

    # Create a conversational chain that uses memory for chat history
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return chain 

def chat_interface():
    """Display the chat interface with chat history"""
    st.subheader("Chat with PSG Documents")
    
    # Add chunk size control in the sidebar
    st.sidebar.subheader("Chat Settings")
    num_chunks = st.sidebar.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=20,
        value=10,
        help="Increase this to get more context for complex questions, decrease for simpler questions"
    )
    
    # Add clear chat history button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        if 'chat_chain' in st.session_state:
            del st.session_state.chat_chain
        st.success("Chat history cleared!")
    
    # Check if settings have changed
    settings_changed = False
    if 'num_chunks' not in st.session_state or st.session_state.num_chunks != num_chunks:
        st.session_state.num_chunks = num_chunks
        settings_changed = True
    
    # Reset chat chain if settings changed
    if settings_changed and 'chat_chain' in st.session_state:
        del st.session_state.chat_chain
    
    # Initialize chat chain if not exists
    if 'chat_chain' not in st.session_state:
        with st.spinner("Setting up chat system..."):
            st.session_state.chat_chain = setup_chat_chain()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the PSG documents"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing PSG documents..."):
                if st.session_state.chat_chain:
                    try:
                        # Get response from chain
                        response = st.session_state.chat_chain({"question": prompt})
                        
                        # Get answer and source documents
                        answer = response.get("answer", "")  # Changed from "result" to match the memory output_key
                        source_docs = response.get("source_documents", [])
                        
                        # Store the current response and sources in session state
                        st.session_state.current_response = answer
                        st.session_state.current_sources = source_docs
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                        # Display the answer
                        st.write(answer)
                        
                        # Display references
                        if source_docs:
                            st.subheader("References")
                            # Create a list of dictionaries for the table
                            ref_data = []
                            for i, source in enumerate(source_docs):
                                # Remove keywords collection
                                ref_data.append({
                                    "Chunk": f"Retrieved Text Chunk {i+1}",
                                    "File Name": source.metadata['filename'],
                                    "Page": source.metadata['page'],
                                    "Active Ingredient": source.metadata.get('Active Ingredient (link to Specific Guidance)', 'N/A'),
                                    "Dosage Form": source.metadata.get('Dosage Form', 'N/A'),
                                    "URL": source.metadata.get('URL', 'N/A')
                                    # Keywords column removed
                                })
                            # Create DataFrame and set Chunk as index
                            df = pd.DataFrame(ref_data)
                            df = df.set_index('Chunk')
                            # Display as a table
                            st.table(df)
                    except Exception as e:
                        st.error(f"Error getting response: {str(e)}")
                        st.info("Please try rephrasing your question or try again later.")
                else:
                    st.error("Chat system is not properly initialized. Please check if preprocess_docs.py has been run.")

def main():
    # Sidebar for mode selection
    st.sidebar.header("Mode Selection")
    
    # Only show normal modes
    mode = st.sidebar.radio("Choose Mode", ["Keyword Search", "Chat"])
    
    if mode == "Keyword Search":
        # Keyword search interface with requested improvements
        st.sidebar.header("Search Options")
        
        # Text input for keywords
        keywords_input = st.sidebar.text_area(
            "Enter keywords (separate multiple keywords with semicolons)",
            height=150,
            help="Example: study design; population; oral tablet",
            placeholder="Enter search terms here..."
        )
        
        # Button to trigger search
        search_button = st.sidebar.button("Search PSGs", type="primary")
        
        # Initialize session state for storing index
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.pdf_contents = {}
            st.session_state.all_pdf_files = []
        
        # First check if the index file exists - this is the preferred way
        if os.path.exists(INDEX_FILE):
            print(f"Found index file: {INDEX_FILE}")
            # Load from index file if it exists
            with st.spinner("Loading indexed PSG data..."):
                index_data = load_index(INDEX_FILE)
                if index_data:
                    pdf_contents = index_data.get("contents", {})
                    all_pdf_files = index_data.get("files", [])
                    st.success(f"Successfully loaded index with {len(all_pdf_files)} PSG files")
                    
                    # Store in session state
                    st.session_state.pdf_contents = pdf_contents
                    st.session_state.all_pdf_files = all_pdf_files
                    
                    # Display total number of PDFs
                    st.sidebar.write(f"Total PSGs in collection: {len(st.session_state.all_pdf_files)}")
                    
                    # Display last index date
                    index_time = datetime.fromtimestamp(os.path.getmtime(INDEX_FILE))
                    st.sidebar.write(f"Index last updated: {index_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.error("Failed to load index file. The file may be corrupted.")
                    return
        # Only check for PDF folder if index file doesn't exist
        elif os.path.exists(PDF_FOLDER):
            print(f"Index file not found, but PDF folder exists: {PDF_FOLDER}")
            # Initialize or load the PDF index from the PDF folder
            with st.spinner("Processing PDF files..."):
                pdf_contents, all_pdf_files = index_pdfs(PDF_FOLDER, force_reindex=False)
                st.session_state.pdf_contents = pdf_contents
                st.session_state.all_pdf_files = all_pdf_files
                
                # Display total number of PDFs
                st.sidebar.write(f"Total PSGs in collection: {len(st.session_state.all_pdf_files)}")
        else:
            # Neither index file nor PDF folder found
            print("Files in current directory:")
            for file in os.listdir("."):
                print(f"  - {file}")
                
            st.error(f"Neither PDF folder '{PDF_FOLDER}' nor index file '{INDEX_FILE}' found.")
            st.info("Please provide either PDF files or a pre-processed index file.")
            return
        
        # Main content area - perform search if button clicked
        if search_button and keywords_input:
            # Split keywords by semicolon and strip whitespace
            keywords = [k.strip() for k in keywords_input.split(';') if k.strip()]
            
            if not keywords:
                st.warning("Please enter at least one keyword.")
                return
            
            st.subheader(f"Searching for PSG files containing all keywords:")
            for i, keyword in enumerate(keywords):
                st.write(f"{i+1}. '{keyword}'")
            
            # Perform search - always case insensitive now
            with st.spinner("Searching..."):
                results = search_pdfs(keywords, st.session_state.pdf_contents, case_sensitive=False)
            
            # Display results
            display_results(results, keywords)
        else:
            # Default view - show instructions
            if not search_button:
                st.info("Enter keywords in the sidebar and click 'Search PSGs' to begin.")
            elif not keywords_input:
                st.warning("Please enter at least one keyword to search.")
    
    elif mode == "Chat":
        chat_interface()

if __name__ == "__main__":
    main() 