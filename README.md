# ChatPSG

ChatPSG is an application that allows you to search for keywords across Product Specific Guidance (PSG) PDF files and chat with an AI about their content.

## Features

- **Keyword Search**: Search across PSG documents using multiple keywords
- **Chat Interface**: Ask questions about PSG content with AI-powered responses
- **Chat History**: Maintains conversation context for natural dialogue
- **Reference Display**: Shows the source documents for each answer

## How to Use

### Keyword Search Mode

1. Select "Keyword Search" from the sidebar
2. Enter keywords separated by semicolons
3. Click "Search PSGs" to find matching documents
4. View results and download as CSV if needed

### Chat Mode

1. Select "Chat" from the sidebar
2. Type your question in the input box
3. View AI's response and source references
4. Continue the conversation with follow-up questions
5. Use "Clear Chat History" button to start a new conversation

## Setup for Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place PSG PDF files in the `pdfs` folder
4. Run the preprocessing script: `python preprocess_docs.py`
5. Run the app: `streamlit run app8.py`

## Deployment

This app can be deployed on Streamlit Cloud by connecting to this GitHub repository.

## Note

You will need to provide your own Google API key for Gemini to work. 