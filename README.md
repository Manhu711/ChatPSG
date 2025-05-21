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
5. Add your Google API key to `.streamlit/secrets.toml` (see API Key Setup below)
6. Run the app: `streamlit run app.py`

## API Key Setup

This application requires a Google Gemini API key to function properly. As the application administrator, you'll need to provide your own API key. You can get one at [Google AI Studio](https://makersuite.google.com/app/apikey).

### For Local Development

Add your API key to `.streamlit/secrets.toml`:

```toml
[api_keys]
GOOGLE_API_KEY = "your-api-key-here"
```

### For Streamlit Cloud Deployment

In the Streamlit Cloud dashboard, go to your app settings, find the "Secrets" section, and add:

```toml
[api_keys]
GOOGLE_API_KEY = "your-api-key-here"
```

## Deployment

This app can be deployed on Streamlit Cloud by connecting to this GitHub repository. Make sure to set up your API key in the Streamlit secrets as described above.

## Note

This application uses the Google Gemini API for AI chat functionality. The API key is managed by the application administrator, so end users don't need to provide their own keys. 