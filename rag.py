# Install necessary packages
# !pip install llama-index llama-index-llms-huggingface llama-index-embeddings-huggingface transformers accelerate bitsandbytes llama-index-readers-web

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from IPython.display import Markdown, display

# Step 1: Scrape “Luke Skywalker” wiki page
# Define the URL of the Wikipedia page
URL = "https://en.wikipedia.org/wiki/Luke_Skywalker"

# Load the BeautifulSoupWebReader to scrape the web page
from llama_index.core import download_loader
BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=[URL])

# Step 2: Chunk it, store it in any vector database like Faiss.
# Configure the HuggingFaceLLM with quantization for efficiency
import torch
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Initialize the LLM with a specific model
Settings.llm = HuggingFaceLLM(
    model_name="berkeley-nest/Starling-LM-7B-alpha",
    tokenizer_name="berkeley-nest/Starling-LM-7B-alpha",
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.8, "do_sample": True},
    device_map="auto",
)

# Step 3: Add any LLM API calling functionality like openai/gemini/claude/groq/deep infra etc.
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Step 4: Create a VectorStoreIndex for chunking and storing the data
from llama_index.core import VectorStoreIndex

Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Create the vector index from the documents
vector_index = VectorStoreIndex.from_documents(
    documents,
)

# Persist the vector index for future use
# vector_index.storage_context.persist(persist_dir='/content')

# Step 5: From question, retrieve top 3 relevant chunks
# Step 6: Pass actual questions with retrieved chunks to LLM.
# Create a query engine to find the top 3 relevant chunks
query_engine = vector_index.as_query_engine(similarity_top_k=3)

# Example query to the engine, You can type your query here

text=input("Input your query here about Luke Skywalker: ")

response = query_engine.query(text)
print(response)
# Step 7: Give an answer.
# Display the answer in markdown format
# display(Markdown(f"<b>{response}</b>"))