import os
import gradio as gr
from dotenv import load_dotenv 
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse

# Load variables from .env file
load_dotenv()

# API keys
llama_cloud_key = os.environ.get("LLAMA_CLOUD_API_KEY")
groq_key = os.environ.get("GROQ_API_KEY")
cohere_key = os.environ.get("COHERE_API_KEY")
if not (llama_cloud_key and groq_key and cohere_key):
    raise ValueError(
        "API Keys not found! Ensure they are passed to the Docker container."
    )

# models name
llm_model_name = "llama3-70b-8192"
embed_model_name = "embed-english-v3.0"

# Global variable for the vector index
vector_index = None

# Initialize the parser
parser = LlamaParse(api_key=llama_cloud_key, result_type="markdown")

# Define file extractor with various common extensions
file_extractor = {
    ".pdf": parser,
    ".docx": parser,
    ".doc": parser,
    ".txt": parser,
    ".csv": parser,
    ".xlsx": parser,
    ".pptx": parser,
    ".html": parser,
    ".jpg": parser,
    ".jpeg": parser,
    ".png": parser,
    ".webp": parser,
    ".svg": parser,
}

# Initialize the Cohere embedding model
embed_model = CohereEmbedding(api_key=cohere_key, model_name=embed_model_name)

# Initialize the LLM
llm = Groq(model="llama3-70b-8192", api_key=groq_key)


# File processing function
def load_files(file_path: str):
    global vector_index
    if not file_path:
        return "No file path provided. Please upload a file."
    
    valid_extensions = ', '.join(file_extractor.keys())
    if not any(file_path.endswith(ext) for ext in file_extractor):
        return f"The parser can only parse the following file types: {valid_extensions}"

    document = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
    vector_index = VectorStoreIndex.from_documents(document, embed_model=embed_model)
    
    print(f"Parsing completed for: {file_path}")
    filename = os.path.basename(file_path)
    return f"Ready to provide responses based on: {filename}"


# Respond function
def respond(message, history):
    global vector_index
    if vector_index is None:
        yield "Please upload a file first to begin the chat."
        return

    try:
        # Create a stateless query engine for each response
        query_engine = vector_index.as_query_engine(streaming=True, llm=llm)
        streaming_response = query_engine.query(message)
        
        # Stream the text response
        partial_text = ""
        for token in streaming_response.response_gen:
            partial_text += token
            # Yield an empty string to cleanup the message textbox and the updated conversation history 
            yield partial_text
    except Exception as e:
        print(f"An error occurred during chat: {e}")
        yield "An error occurred while processing your request. Please try again."


# Clear function
def clear_state():
    global vector_index
    vector_index = None
    return [None, None, None]


# UI Setup
with gr.Blocks(
    theme=gr.themes.Monochrome(
        primary_hue="indigo",
        secondary_hue="blue",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
    css="footer {visibility: hidden}",
) as demo:
    gr.Markdown("# Document Q&A 🤖📃")
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Controls")
            file_input = gr.File(
                file_count="single", type="filepath", label="Upload Document"
            )
            output = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                btn = gr.Button("1. Process Document", variant="primary", scale=2)
                clear = gr.Button("Clear All", scale=1)

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(
                    height=500,
                    label="Chat Window",
                ),
                textbox=gr.Textbox(
                    placeholder="2. Ask questions about the document here...",
                    container=False,
                    scale=7,
                ),
                submit_btn="Ask",
                show_progress="full",
            )

    # Set up Gradio interactions
    btn.click(fn=load_files, inputs=file_input, outputs=output)
    
    clear.click(
        fn=clear_state,  # Use the clear_state function 
        outputs=[file_input, output, chatbot],
        queue=False 
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch()
