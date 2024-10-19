import os

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.perplexity import Perplexity
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]

# Set up Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Set up HuggingFace embeddings
embed_model = HuggingFaceEmbedding(
    model_name="Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
)

# Set global settings for LLM to be Perplexity to avoid LlamaIndex looking for OpenAI key
# Reference: https://stackoverflow.com/questions/76771761/why-does-llama-index-still-require-an-openai-key-when-using-hugging-face-local-e
llm = Perplexity(
    api_key=PERPLEXITY_API_KEY, model="llama-3.1-70b-instruct", temperature=0.5
)
Settings.llm = llm

# Create vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Create vector store index
index = VectorStoreIndex.from_vector_store(
    vector_store, embed_model=embed_model, llm=llm
)

# Create query engine
query_engine = index.as_query_engine()


def answer_query(user_query):
    # Step 1: Accept user query (already done by passing it as an argument)

    # Steps 2-4: Find embeddings, determine context, and use ChatGPT to answer
    response = query_engine.query(user_query)

    return response.response


while True:
    user_input = input("Enter your query: ")
    answer = answer_query(user_input)
    print("Answer:", answer)
