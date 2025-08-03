# Document Q&A with LLMs, Docker, and Hugging Face

This project is a web-based application that allows you to chat with your documents. You can upload a document (PDF, DOCX, TXT, etc.), and the application will process it to answer your questions based on its content.

The application is built with:

*   **Backend:** Python, LlamaIndex, Groq, and Cohere.
*   **Frontend:** Gradio for the user interface.
*   **Containerization:** Docker for easy deployment.

## How it Works

1.  **Document Parsing:** When you upload a document, it's parsed using LlamaParse to extract the text content.
2.  **Embeddings:** The extracted text is then converted into vector embeddings using Cohere's embedding model.
3.  **LLM Interaction:** When you ask a question, the application uses the Groq API (with Llama 3) to find the most relevant information in the document and generate a response.

## Running the Application with Docker

### Prerequisites

*   Docker installed on your machine.
*   API keys for:
    *   LlamaParse (LLAMA_CLOUD_API_KEY)
    *   Groq (GROQ_API_KEY)
    *   Cohere (COHERE_API_KEY)

### Steps

1.  **Build the Docker Image:**

    ```bash
    docker build -t document-qa .
    ```

2.  **Run the Docker Container:**

    Replace `your_llama_cloud_key`, `your_groq_key`, and `your_cohere_key` with your actual API keys.

    ```bash
    docker run -p 7860:7860 \
      -e LLAMA_CLOUD_API_KEY="your_llama_cloud_key" \
      -e GROQ_API_KEY="your_groq_key" \
      -e COHERE_API_KEY="your_cohere_key" \
      document-qa
    ```

3.  **Access the Application:**

    Open your web browser and go to `http://localhost:7860`.

## Deploying to Hugging Face Spaces

You can deploy this application to Hugging Face Spaces directly from this repository.

### Steps

1.  **Create a new Hugging Face Space:**
    *   Go to [huggingface.co/new-space](https://huggingface.co/new-space).
    *   Give your Space a name.
    *   Select **Docker** as the Space SDK.
    *   Choose "Docker from scratch".
    *   Create the Space.

2.  **Upload the files:**
    *   Upload `app.py`, `requirements.txt`, and `Dockerfile` to your Hugging Face Space repository.

3.  **Add Secrets:**
    *   In your Space's settings, go to the **Secrets** section.
    *   Add the following secrets with your API keys:
        *   `LLAMA_CLOUD_API_KEY`
        *   `GROQ_API_KEY`
        *   `COHERE_API_KEY`

4.  **Deploy:**
    *   Hugging Face will automatically build the Docker image from your `Dockerfile` and deploy the application. Once the build is complete, your application will be live.

```