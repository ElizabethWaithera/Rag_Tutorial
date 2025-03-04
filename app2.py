import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
from dotenv import load_dotenv
import textwrap
import io
import tempfile

from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated imports for newer OpenAI client compatibility
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain

# Load environment variables from .env file
os.environ.clear()
load_dotenv(override=True)


if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key not found in environment variables. Please enter your API key:", type="password")
    if not api_key:
        st.warning("Please enter an OpenAI API key to use this application.")
        st.stop()

# Constants
DATA_DIR = "knowledge_base"
CHROMA_DIR = "chroma_db"
SAMPLE_DOCS_DIR = "sample_docs"

# Ensure directories exist
for directory in [DATA_DIR, SAMPLE_DOCS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Setup page config
st.set_page_config(
    page_title="RAG Demo: Compare LLM vs RAG Responses",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved appearance
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; margin-bottom: 1rem;}
    .sub-header {font-size: 1.8rem; margin-bottom: 0.8rem;}
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .standard-box {
        background-color: #f0f2f6;
        border-left: 0.5rem solid #9e9e9e;
    }
    .rag-box {
        background-color: #e1f5fe;
        border-left: 0.5rem solid #039be5;
    }
    .source-box {
        background-color: #f5f5f5;
        border-left: 0.3rem solid #4caf50;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    .stButton button {width: 100%;}
    .metadata {font-size: 0.8rem; color: #666;}
    .highlight {background-color: #ffff00; padding: 0 0.2rem;}
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>🔍 RAG Demo: Standard LLM vs. RAG-Enhanced Responses</h1>", unsafe_allow_html=True)

st.markdown("""
This application demonstrates the power of **Retrieval-Augmented Generation (RAG)** by comparing:

1. **Standard LLM Responses**: Using only the model's pre-trained knowledge
2. **RAG-Enhanced Responses**: Augmented with information from a custom knowledge base

Try asking questions to see the difference!
""")

# Function to create sample documents if they don't exist
def create_sample_documents():
    # Sample news articles with recent information
    sample_articles = [
        {
            "title": "Breakthrough in Quantum Computing Announced",
            "date": "2024-01-15",
            "content": """
Researchers at Quantum Future Labs have announced a major breakthrough in quantum computing stability.
Their new approach allows qubits to maintain coherence for up to 5 minutes, a dramatic improvement over
previous records measured in milliseconds. Dr. Sarah Chen, lead researcher on the project, stated,
"This development opens the door to practical quantum computing applications that were previously
theoretical." The team achieved this by developing a novel cooling system that nearly eliminates
thermal interference. Industry experts project this could accelerate the timeline for quantum
advantage by 3-5 years. The research was published in the March 2024 issue of Quantum Science Review.
            """
        },
        {
            "title": "New Climate Agreement Reached at Global Summit",
            "date": "2023-11-30",
            "content": """
Representatives from 198 countries reached a landmark climate agreement yesterday at the Global
Climate Summit in Nairobi. The agreement, called the "Nairobi Protocol," commits signatories to
reducing carbon emissions by 45% before 2035 compared to 2010 levels. The protocol also establishes
a $200 billion climate adaptation fund for developing nations, with major economies providing the
majority of financing. Unlike previous agreements, the Nairobi Protocol includes enforceable penalties
for nations that fail to meet their targets. Environmental groups have cautiously praised the agreement,
while noting that implementation will be crucial. The first review of progress will occur in 2026.
            """
        },
        {
            "title": "Revolutionary Battery Technology Extends EV Range",
            "date": "2024-02-08",
            "content": """
Electric vehicle manufacturer NeoVolt has unveiled a new solid-state battery technology that promises
to extend the range of electric vehicles by up to 70%. The new batteries, which will enter production
in late 2024, can fully charge in under 15 minutes and maintain 90% capacity after 2,000 charge cycles.
"This addresses the two biggest concerns consumers have about EVs: range anxiety and charging time,"
said NeoVolt CEO Michael Rodriguez. The company's stock jumped 28% following the announcement. The
technology was developed in partnership with researchers from Stanford University and uses a novel
lithium-ceramic composite that eliminates the risk of thermal runaway that plagued earlier batteries.
Industry analysts predict this could accelerate EV adoption significantly in the coming years.
            """
        },
        {
            "title": "Major Breakthrough in Alzheimer's Treatment",
            "date": "2023-12-12",
            "content": """
Pharmaceutical company Neurova has reported promising results from Phase III clinical trials of their
new Alzheimer's treatment, Memoclear. The drug showed a 47% reduction in cognitive decline compared
to placebo in patients with early to moderate Alzheimer's disease. Unlike previous treatments that
focused on amyloid plaques, Memoclear targets tau protein tangles and neuroinflammation simultaneously.
Dr. James Wong, principal investigator for the trial, called the results "the most significant advance
in Alzheimer's treatment in decades." The 18-month trial involved 2,800 patients across 24 countries.
Neurova has filed for FDA approval and expects a decision by August 2024. If approved, the treatment
could be available to patients by early 2025. The company is already planning trials to determine if
the drug could be effective as a preventative measure for those at high risk of developing Alzheimer's.
            """
        },
        {
            "title": "Global Economic Forum Predicts Significant Shift in Market Dynamics",
            "date": "2024-01-22",
            "content": """
The 2024 Global Economic Forum concluded yesterday with economists predicting a significant shift in
global market dynamics over the next decade. The consensus forecast suggests that emerging economies,
particularly in Southeast Asia and Africa, will account for over 50% of global growth by 2030. "We're
witnessing a historic rebalancing of economic power," noted Dr. Elena Kapoor, chief economist at World
Financial Institute. The forum highlighted several factors driving this shift, including demographic
advantages, technological leapfrogging, and strategic investments in green infrastructure. The report
also warned of potential challenges, including geopolitical tensions and climate-related disruptions.
Recommended policy responses include strengthening international financial institutions, improving
global coordination on digital currencies, and developing more resilient supply chains. The forum's
complete economic outlook will be published next month.
            """
        }
    ]
    
    # Write sample articles to files
    for i, article in enumerate(sample_articles):
        file_path = os.path.join(SAMPLE_DOCS_DIR, f"article_{i+1}.txt")
        with open(file_path, "w") as f:
            f.write(f"Title: {article['title']}\n")
            f.write(f"Date: {article['date']}\n\n")
            f.write(textwrap.dedent(article['content']))
    
    return len(sample_articles)

# Function to load documents and create vector store
@st.cache_resource(show_spinner=False)
def create_vector_store(use_sample_docs=True, custom_docs=None):
    documents = []
    
    try:
        # Load sample documents if requested
        if use_sample_docs:
            if not os.path.exists(SAMPLE_DOCS_DIR) or len(os.listdir(SAMPLE_DOCS_DIR)) == 0:
                create_sample_documents()
            
            # First check if directory exists and has files
            if os.path.exists(SAMPLE_DOCS_DIR) and len(os.listdir(SAMPLE_DOCS_DIR)) > 0:
                try:
                    sample_loader = DirectoryLoader(SAMPLE_DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
                    sample_docs = sample_loader.load()
                    documents.extend(sample_docs)
                    st.session_state.document_sources = [f"{SAMPLE_DOCS_DIR}/{f}" for f in os.listdir(SAMPLE_DOCS_DIR)]
                    st.info(f"Loaded {len(sample_docs)} sample documents")
                except Exception as e:
                    st.error(f"Error loading sample documents: {str(e)}")
                    # Create samples again to ensure they exist
                    create_sample_documents()
        
        # Load custom documents if provided
        if custom_docs:
            for doc_path in custom_docs:
                try:
                    # Check file extension
                    file_extension = os.path.splitext(doc_path)[1].lower()
                    
                    if file_extension == '.pdf':
                        loader = PyPDFLoader(doc_path)
                    else:
                        loader = TextLoader(doc_path)
                        
                    custom_doc = loader.load()
                    documents.extend(custom_doc)
                    # Add to document sources if not already present
                    if doc_path not in st.session_state.document_sources:
                        st.session_state.document_sources.append(doc_path)
                except Exception as e:
                    st.error(f"Error loading document {doc_path}: {str(e)}")
        
        # Check if we have documents to process
        if not documents:
            st.warning("No documents loaded. Please upload or enable sample documents.")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store - updated for newer OpenAI client
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Log for debugging
        st.info(f"Creating vector store with {len(chunks)} document chunks")
        
        # Create and persist vector store
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Function to query LLM with and without RAG
def query_llm(question, use_rag=True, temperature=0.1, model_name="gpt-4o-mini"):
    start_time = time.time()
    
    try:
        # Initialize LLM with the specified model - updated for newer OpenAI client
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key  # Changed from openai_api_key to api_key
        )
        
        if use_rag:
            # Get or create vector store
            with st.spinner("Processing knowledge base..."):
                vector_store = create_vector_store(
                    use_sample_docs=st.session_state.use_sample_docs,
                    custom_docs=st.session_state.custom_docs
                )
            
            if vector_store is None:
                return "No knowledge base available. Please upload documents or enable sample documents.", [], time.time() - start_time
            
            # Create retriever with increased k for better recall
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Increased from 3 to 5 for better context
            )
            
            # Improved RAG Prompt Template
            template = """
            You are a helpful assistant that provides accurate information based on the provided context.
            Use ONLY the following context to answer the question. If the context doesn't contain enough 
            information to provide a complete answer, acknowledge the limitations of what you know from 
            the provided context. Do not make up or infer information that's not supported by the context.
            
            CONTEXT:
            {context}
            
            QUESTION: {question}
            
            ANSWER:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create and run RAG chain
            with st.spinner("Retrieving relevant documents..."):
                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )
            
            # Get response
            with st.spinner("Generating RAG-enhanced response..."):
                result = rag_chain({"query": question})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
            
            elapsed_time = time.time() - start_time
            return answer, source_docs, elapsed_time
            
        else:
            # Standard LLM Prompt Template
            template = """
            You are a helpful assistant that provides information based on your training.
            
            QUESTION: {question}
            
            ANSWER:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["question"]
            )
            
            # Create and run standard chain
            with st.spinner("Generating standard response..."):
                chain = LLMChain(llm=llm, prompt=prompt)
                answer = chain.run(question=question)
            
            elapsed_time = time.time() - start_time
            return answer, [], elapsed_time
    
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        return error_message, [], time.time() - start_time

# Initialize session state
if 'use_sample_docs' not in st.session_state:
    st.session_state.use_sample_docs = True
if 'custom_docs' not in st.session_state:
    st.session_state.custom_docs = []
if 'document_sources' not in st.session_state:
    st.session_state.document_sources = []
if 'last_question' not in st.session_state:
    st.session_state.last_question = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = "gpt-4o-mini"  # Default to GPT-4o-mini

# Sidebar for configuration
with st.sidebar:
    st.header("📋 Configuration")
    
    # API Key status
    if api_key:
        st.success("✅ OpenAI API Key detected from environment variables")
    else:
        st.error("❌ No OpenAI API Key found. Please set the OPENAI_API_KEY environment variable.")
    
    # Model selection
    st.subheader("Model Selection")
    model_options = [
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4o"
    ]
    selected_model = st.selectbox(
        "Select OpenAI Model:",
        options=model_options,
        index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0
    )
    
    if selected_model != st.session_state.model_name:
        st.session_state.model_name = selected_model
    
    # Knowledge base section
    st.header("📚 Knowledge Base")
    
    # Sample documents toggle
    use_sample = st.checkbox("Use sample documents", value=st.session_state.use_sample_docs)
    if use_sample != st.session_state.use_sample_docs:
        st.session_state.use_sample_docs = use_sample
        # Clear cache to reload documents
        st.cache_resource.clear()
    
    # Upload custom documents
    st.subheader("Upload Custom Documents")
    uploaded_files = st.file_uploader("Upload documents:", 
                                     type=["txt", "md", "pdf"], 
                                     accept_multiple_files=True)
    
    if uploaded_files:
        save_docs = st.button("Add to Knowledge Base")
        if save_docs:
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    # Get file extension
                    file_extension = os.path.splitext(file.name)[1].lower()
                    
                    # Save uploaded file
                    file_path = os.path.join(DATA_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Add to custom docs if not already there
                    if file_path not in st.session_state.custom_docs:
                        st.session_state.custom_docs.append(file_path)
                
                # Clear cache to reload documents
                st.cache_resource.clear()
                st.success(f"Added {len(uploaded_files)} documents to knowledge base!")
    
    # Show current knowledge base
    st.subheader("Current Knowledge Base")
    if st.session_state.use_sample_docs:
        st.info(f"Using sample documents from {SAMPLE_DOCS_DIR}")
        
        # Create samples if they don't exist yet
        if not os.path.exists(SAMPLE_DOCS_DIR) or len(os.listdir(SAMPLE_DOCS_DIR)) == 0:
            with st.spinner("Creating sample documents..."):
                num_created = create_sample_documents()
                st.success(f"Created {num_created} sample documents")
    
    if st.session_state.custom_docs:
        st.write("Custom documents:")
        for i, doc_path in enumerate(st.session_state.custom_docs):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(os.path.basename(doc_path))
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.custom_docs.remove(doc_path)
                    st.cache_resource.clear()
                    st.experimental_rerun()
    elif not st.session_state.use_sample_docs:
        st.warning("No documents in knowledge base. Please upload documents or enable sample documents.")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        
        # Add rebuild knowledge base button
        if st.button("Rebuild Knowledge Base"):
            st.cache_resource.clear()
            with st.spinner("Rebuilding knowledge base..."):
                create_vector_store(
                    use_sample_docs=st.session_state.use_sample_docs,
                    custom_docs=st.session_state.custom_docs
                )
            st.success("Knowledge base rebuilt successfully!")

# Create tabs for main interface (removed the "How RAG Works" tab)
main_tab, explorer_tab = st.tabs([
    "🔍 RAG Demo", "📚 Knowledge Base Explorer"
])

# Main RAG Demo tab
with main_tab:
    # Query input section
    st.markdown("<h2 class='sub-header'>Ask a Question</h2>", unsafe_allow_html=True)
    
    # Example questions
    with st.expander("Example Questions"):
        example_questions = [
            "What was the breakthrough announced in quantum computing?",
            "Tell me about the recent climate agreement.",
            "What is the new battery technology for electric vehicles?",
            "What are the latest developments in Alzheimer's treatment?",
            "What did economists predict at the Global Economic Forum?"
        ]
        
        for i, example in enumerate(example_questions):
            if st.button(f"Try this: {example}", key=f"example_{i}"):
                st.session_state.question = example
                st.session_state.run_query = True
    
    # Question input
    col1, col2 = st.columns([3, 1])
    with col1:
        # Fix the widget key conflict by not using session state directly in the value
        current_question = st.session_state.get("question", "")
        question = st.text_area("Enter your question:", 
                              value=current_question,
                              height=80, 
                              key="question_input")  # Changed key to avoid conflict
        
        # Update session state after widget is created
        st.session_state.question = question
    with col2:
        st.write("")
        st.write("")
        run_query = st.button("🔍 Ask Question", use_container_width=True)
        clear_input = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_input:
            st.session_state.question = ""
            st.session_state.run_query = False
            # Use experimental rerun to clear the UI
            st.experimental_rerun()
    
    # Store run_query in session state to handle rerun
    if run_query:
        st.session_state.run_query = True
    
    # Process query if button was clicked or if question is in session state
    if st.session_state.get("run_query", False) and question:
        # Check if it's a new question
        if st.session_state.last_question != question:
            st.session_state.last_question = question
            
            # Get model name from session state
            model_name = st.session_state.model_name
            
            # Get standard LLM response and RAG response simultaneously
            standard_response, _, standard_time = query_llm(
                question, 
                use_rag=False, 
                temperature=temperature,
                model_name=model_name
            )
            
            rag_response, source_docs, rag_time = query_llm(
                question, 
                use_rag=True, 
                temperature=temperature,
                model_name=model_name
            )
            
            # Display responses
            st.markdown("<h2 class='sub-header'>Results Comparison</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3>Standard LLM Response</h3>", unsafe_allow_html=True)
                st.markdown("<div class='info-box standard-box'>", unsafe_allow_html=True)
                st.write(standard_response)
                st.markdown(f"<p class='metadata'>Response time: {standard_time:.2f} seconds | Model: {model_name}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<h3>RAG-Enhanced Response</h3>", unsafe_allow_html=True)
                st.markdown("<div class='info-box rag-box'>", unsafe_allow_html=True)
                st.write(rag_response)
                st.markdown(f"<p class='metadata'>Response time: {rag_time:.2f} seconds | Model: {model_name}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display sources
            if source_docs:
                st.markdown("<h3>Sources Used for RAG Response</h3>", unsafe_allow_html=True)
                
                for i, doc in enumerate(source_docs):
                    with st.expander(f"Source {i+1}"):
                        st.markdown("<div class='source-box'>", unsafe_allow_html=True)
                        st.markdown(doc.page_content)
                        
                        # Show metadata if available
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown("<p class='metadata'>Metadata:</p>", unsafe_allow_html=True)
                            for key, value in doc.metadata.items():
                                st.markdown(f"<p class='metadata'><b>{key}:</b> {value}</p>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
    
    # Welcome message for first-time users
    elif not question:
        st.markdown("""
        <div class='info-box'>
        <h3>Welcome to the RAG Demo!</h3>
        
        <p>This demo shows how Retrieval-Augmented Generation (RAG) improves AI responses by connecting language models to a knowledge base.</p>
        
        <p>To get started:</p>
        <ol>
            <li>Type a question in the text box above or try one of the example questions</li>
            <li>Click "Ask Question" to see both standard and RAG-enhanced responses</li>
            <li>Compare the differences between the two approaches</li>
        </ol>
        
        <p>Try asking about topics covered in our knowledge base, such as recent science breakthroughs, climate agreements, or economic developments!</p>
        </div>
        """, unsafe_allow_html=True)

# Knowledge Base Explorer tab
with explorer_tab:
    st.markdown("<h2 class='sub-header'>Knowledge Base Contents</h2>", unsafe_allow_html=True)
    
    # Initialize knowledge base if needed
    if st.session_state.use_sample_docs and (not os.path.exists(SAMPLE_DOCS_DIR) or len(os.listdir(SAMPLE_DOCS_DIR)) == 0):
        with st.spinner("Creating sample documents..."):
            create_sample_documents()
    
    # Collect documents to display
    all_docs = []
    
    # Process document sources
    for source in st.session_state.document_sources:
        try:
            if os.path.exists(source):
                # Handle different file formats
                file_extension = os.path.splitext(source)[1].lower()
                
                if file_extension == '.pdf':
                    try:
                        loader = PyPDFLoader(source)
                        pages = loader.load()
                        
                        title = os.path.basename(source)
                        date = "Unknown"
                        content = "\n\n".join([page.page_content for page in pages])
                        
                        all_docs.append({
                            "title": title,
                            "date": date,
                            "content": content,
                            "source": source
                        })
                    except Exception as e:
                        st.warning(f"Error loading PDF {source}: {str(e)}")
                else:
                    with open(source, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    
                    # Try to extract title and date
                    title = os.path.basename(source)
                    date = "Unknown"
                    
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if line.startswith("Title:"):
                            title = line[6:].strip()
                        if line.startswith("Date:"):
                            date = line[5:].strip()
                    
                    all_docs.append({
                        "title": title,
                        "date": date,
                        "content": content,
                        "source": source
                    })
        except Exception as e:
            st.warning(f"Error loading document {source}: {str(e)}")
    
    if not all_docs:
        st.warning("No documents found in the knowledge base. Please upload documents or enable sample documents.")
    else:
        # Display documents
        st.write(f"Found {len(all_docs)} documents in the knowledge base:")
        
        for i, doc in enumerate(all_docs):
            with st.expander(f"{doc['title']} ({doc['date']})"):
                st.markdown("<div class='source-box'>", unsafe_allow_html=True)
                st.markdown(f"**Source:** {doc['source']}")
                st.markdown("---")
                st.markdown(doc["content"])
                st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
<p>RAG Demo Application | Created for educational purposes</p>
<p><small>Powered by LangChain, OpenAI, and Streamlit</small></p>
</div>
""", unsafe_allow_html=True)

# Dependencies info
with st.sidebar.expander("Dependencies Information"):
    st.markdown("""
    ### Required Packages
    
    To run this application, you need:
    
    ```
    pip install streamlit langchain langchain-openai chromadb openai tiktoken python-dotenv
    ```
    
    Note: Make sure you have OpenAI SDK version 1.0.0 or newer installed.
    """)

# Run the app
# To run: streamlit run app.py
