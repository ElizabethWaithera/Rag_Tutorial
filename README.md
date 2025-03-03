# RAG Demo Application

This Streamlit application demonstrates the power of Retrieval-Augmented Generation (RAG) by comparing standard LLM responses with RAG-enhanced responses. It clearly illustrates why RAG matters when dealing with recent information.

## Features

- Side-by-side comparison of standard LLM vs. RAG-enhanced responses
- Built-in sample documents with recent information
- Support for custom document uploads
- Knowledge base explorer to view document contents
- Educational content explaining how RAG works

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag-demo.git
cd rag-demo
```

2. Create a virtual environment and activate it:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root directory
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Use the application:
   - Ask questions about the sample documents or your own uploaded documents
   - Compare the standard LLM response with the RAG-enhanced response
   - Explore the knowledge base to view document contents
   - Learn about how RAG works

## Application Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Required Python packages
- `.env`: Environment variables (add your OpenAI API key here)
- `knowledge_base/`: Directory for storing uploaded documents
- `sample_docs/`: Directory for sample documents
- `chroma_db/`: Directory for the Chroma vector database

## How It Works

This application demonstrates Retrieval-Augmented Generation (RAG), which enhances language model outputs by:

1. Creating a knowledge base of documents
2. Converting documents into embeddings (vector representations)
3. Storing these embeddings in a vector database
4. When a question is asked:
   - Converting the question to an embedding
   - Finding the most similar document chunks
   - Providing these chunks as context to the language model
   - Generating a response based on this specific context

This approach addresses two major limitations of large language models:
- **Knowledge cutoff**: LLMs only know information up to their training date
- **Hallucinations**: LLMs can generate plausible but incorrect information

## Sample Documents

The application includes sample documents about:
- A breakthrough in quantum computing
- A new climate agreement
- Revolutionary battery technology for electric vehicles
- Advances in Alzheimer's treatment
- Economic forecasts from a global forum

These documents contain "recent" information that would be beyond a typical LLM's knowledge cutoff date, demonstrating the value of RAG for accessing up-to-date information.

## Custom Documents

You can upload your own text documents to the knowledge base. The application will process these documents and make them available for RAG queries.

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages listed in `requirements.txt`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
