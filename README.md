# PDF RAG Assistant

A Streamlit-based PDF question-answering assistant that uses retrieval-augmented generation (RAG) to answer questions from uploaded documents with relevant source citations.

## Features

- Extracts text from PDF documents.
- Splits documents into searchable chunks.
- Uses local embeddings and ChromaDB for document retrieval.
- Reranks retrieved passages to improve relevance.
- Generates answers using Groq's hosted language model.
- Uses document evidence and source page numbers to support answers.
- Abstains when the available evidence is insufficient.

## Setup

1. Clone the repository and open the project directory.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env`.
5. Add your Groq API key to `.env`.
6. Run the application:

   ```bash
   streamlit run app.py
   ```

## Configuration

The application reads its Groq settings from environment variables.

| Variable | Default |
|---|---|
| `GROQ_API_KEY` | Required |
| `GROQ_MODEL` | `openai/gpt-oss-20b` |
| `GROQ_TIMEOUT_SECONDS` | `30` |
| `GROQ_MAX_RETRIES` | `0` |
| `GROQ_TEMPERATURE` | `0` |
| `GROQ_MAX_OUTPUT_TOKENS` | `512` |

Never commit your `.env` file or expose your API key.

## Deployment

The application can be deployed using Streamlit Community Cloud.

Add the required Groq configuration under the app's **Settings → Secrets** section. Use the same variable names as the local environment configuration.

Do not commit API keys or deployment secrets to GitHub.

## Cost and limitations

The project is designed to use local document processing and the Groq Free Plan for language model inference.

- Keep the Groq account on its Free Plan and do not enable paid usage.
- Free-tier limits are shared and may restrict requests.
- No automatic paid-provider fallback is configured.
- Streamlit Community Cloud resources are limited, and local document indexes may need to be rebuilt after a restart.
- Token counts are estimates, not exact provider tokenizer counts.
