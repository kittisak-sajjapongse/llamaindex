import os

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.perplexity import Perplexity
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from pythainlp import word_tokenize

from cb_handlers import LLMTemplateLogger

load_dotenv()
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]
OLLAMA_BASE_URL = f"http://{os.environ['OLLAMA_BASE_ENDPOINT']}"


class ThaiTextSplitter:
    def split_text(self, text: str) -> list[str]:
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
        return " ".join(tokens)


# ----------------------------------------------------------
# Important note: It is important to set the global callback manager before
# instantiated any LlamaIndex objects. This allows components instantiated
# subsequently to set the manager properly.

# Initialize the custom callback handler
llm_template_logger = LLMTemplateLogger()

# Create a CallbackManager with the custom handler
callback_manager = CallbackManager([llm_template_logger])

Settings.callback_manager = callback_manager

# ----------------------------------------------------------

# Set up Pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Set up HuggingFace embeddings
embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large", trust_remote_code=True
)

# Set global settings for LLM to be Perplexity to avoid LlamaIndex looking for OpenAI key
# Reference: https://stackoverflow.com/questions/76771761/why-does-llama-index-still-require-an-openai-key-when-using-hugging-face-local-e
llm = Perplexity(
    api_key=PERPLEXITY_API_KEY, model="llama-3.1-70b-instruct", temperature=0.5
)
llm_private = Ollama(
    model="llama3.1:70b", base_url=OLLAMA_BASE_URL, request_timeout=600.0
)
Settings.llm = llm_private

# Create vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Create vector store index
index = VectorStoreIndex.from_vector_store(
    vector_store, embed_model=embed_model, llm=llm
)

# Create node retriever
retriever = index.as_retriever()

# Create response synthesizer for querying LLM
response_synthesizer = get_response_synthesizer()


def answer_query(user_query):
    thai_text_splitter = ThaiTextSplitter()
    retrieved_nodes = retriever.retrieve(thai_text_splitter.split_text(user_query))
    response = response_synthesizer.synthesize(query=user_query, nodes=retrieved_nodes)

    # Print retrieved nodes
    for i, node in enumerate(retrieved_nodes):
        print(f"Node: {i}")
        print(f"    Score:    {node.score}")
        print(f"    Metadata: {node.metadata}")

    # Print LLM template for debugging
    messages = llm_template_logger.get_messages()
    for message in messages:
        print(f"<DEBUG> {message.content}")
    return response.response


while True:
    user_input = input("Enter your query: ")
    answer = answer_query(user_input)
    print("Answer:", answer)
