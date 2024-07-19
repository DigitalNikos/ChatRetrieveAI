# ChatRetrieveAI README

## Overview

ChatRetrieveAI is a chatbot application that uses the Retrieval-Augmented Generation (RAG) approach to interact with Documents and external sources like Wikipedia.

## Features

✅ Document Ingestion: Upload and ingest documents or URLs to create a searchable knowledge base.

✅ Documents(.txt .docs urls) Ingestion: TODO

✅ Chat History: TODO

✅ Show metadata to answer(filename, page, url): TODO

✅ Domain: Users are able to choose specific domain for the Chatbot conversation.

✅ Contextual Chat: Engage in conversations with the chatbot, which uses the content of the uploaded PDFs and additional sources like Wikipedia.

✅Streamlit Interface: User-friendly interface built with Streamlit for easy interaction.

## Getting Started

### Prerequisites

- Python: Ensure you have Python 3.8 or higher installed.
- Ollama: Download and install Ollama.

### Installation and Setup

1. #### Download Ollama:
   - Visit the Ollama page and download the software:
     [https://ollama.com](https://ollama.com)
2. #### Pull the Mistral Model:
   - Open a terminal and pull the Mistral model with the following command:
     ```
     ollama pull mistral
     ```
3. #### Clone the Repository:
   - Clone the project repository from GitHub:
   ```
   git clone https://github.com/DigitalNikos/ChatRetrieveAI.git
   ```
   ```
   cd ChatRetrieveAI
   ```
4. #### Create a Virtual Environment:
   - Create a Python virtual environment:
   ```
   python -m venv env
   ```
5. #### Activate the Virtual Environment:
   - On Windows:
   ```
   .\env\Scripts\activate
   ```
   - On macOS and Linux:
   ```
   source env/bin/activate
   ```
6. #### Install the Required Dependencies:
   - Install the dependencies listed in the requirements.txt file:
   ```
   pip install -r requirements.txt
   ```

### Running the Project

1. #### Start the Streamlit Application:
   - Run the Streamlit application with the following command:
   ```
   streamlit run chatbot.py
   ```

## Configuration

The application settings can be modified in the config.py file. Key settings include:

- MODEL: The language model to use.
- SPLITTER_CHUNK_SIZE: The chunk size for splitting documents.
- SPLITTER_CHUNK_OVERLAP: The overlap size for chunks.
- N_DOCUMENTS_TO_RETRIEVE: Number of documents to retrieve for each query.
- RETRIEVER_SCORE_THRESHOLD: Threshold score for the retriever.
