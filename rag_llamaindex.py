import logging
import sys

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


documents = SimpleDirectoryReader("data").load_data()

Settings.llm = Ollama(model="llama3.1:latest", request_timeout=120, temperature=0.01)

# specific settings for nomic embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    text_instruction="search_document:",
    query_instruction="search_query:",
)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=4)

response = query_engine.query("what is nougat and how does it work?")

print(response)
