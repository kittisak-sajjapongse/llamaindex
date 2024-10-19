import os

from dotenv import load_dotenv
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file.unstructured import UnstructuredReader
from llama_index.vector_stores.pinecone import PineconeVectorStore

load_dotenv()
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
IMPORT_DATA_DIR = os.environ["IMPORT_DATA_DIR"]

# Set up Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Load HTML documents from a directory
print("Load HTML documents")
unstructured_reader = UnstructuredReader()
documents = SimpleDirectoryReader(
    IMPORT_DATA_DIR, file_extractor={".html": unstructured_reader}
).load_data()

# Extract only text from the documents
print("Extracting text")
text_documents = [Document(text=doc.text) for doc in documents]

print("Creating SimpleNodeParser")
# Set up node parser for chunking
node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

# Set up HuggingFace embedding
print("Creating embedding model")
embed_model = HuggingFaceEmbedding(
    model_name="Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
)

# Set up Pinecone vector store
print("Create vector store")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index and ingest chunks into Pinecone
print("Starting ingesting")
index = VectorStoreIndex.from_documents(
    text_documents,
    node_parser=node_parser,
    embed_model=embed_model,
    vector_store=vector_store,
    storage_context=storage_context,
)

print("Documents loaded, chunked, and ingested into Pinecone successfully!")
