import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.data_structs import Node
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.perplexity import Perplexity
from llama_index.readers.file.unstructured import UnstructuredReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from pythainlp import word_tokenize

from cb_handlers import LLMTemplateLogger

load_dotenv()
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]
OLLAMA_BASE_URL = f"http://{os.environ['OLLAMA_BASE_ENDPOINT']}"
ORIGINAL_PROMPT_TEMPLATE = (
    "You are an expert Q&A system that is trusted around the world.\n"
    "Always answer the query using the provided context information, and not prior knowledge.\n"
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query:{query_str}\n"
    "Answer:\n"
    "assistant:"
)
CUSTOM_PROMPT_TEMPLATE = (
    "You are an expert Q&A system that is trusted around the world.\n"
    "Always answer the query using the provided context information, and not prior knowledge.\n"
    "Please extract information from the given list of JSON objects as much as possible.\n"
    "List of JSON objects:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the list of JSON objects, do the following:\n"
    "1. Answer the query below without using prior knowledge by examining the 'text' key of all JSON objects in the list.\n"
    "2. Report the values of the 'source' key of all JSON objects to the user \n"
    "Query:{query_str}\n"
    "Answer:\n"
    "assistant:"
)


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

# Create prompt template
prompt_template = PromptTemplate(ORIGINAL_PROMPT_TEMPLATE)

# Create response synthesizer for querying LLM
response_synthesizer = get_response_synthesizer(
    response_mode="compact",
    text_qa_template=prompt_template,
)


def answer_query(user_query):
    thai_text_splitter = ThaiTextSplitter()
    retrieved_nodes = retriever.retrieve(thai_text_splitter.split_text(user_query))

    # Prepare context with source metadata
    file_set = set()
    for node in retrieved_nodes:
        source = node.metadata.get("source", "Unknown source")
        file_set.add(source)

    processed_nodes = []
    unstructured_reader = UnstructuredReader()
    for file in file_set:
        file_path = Path(f"./dataset/{file}")
        documents = unstructured_reader.load_data(file=file_path)
        doc_str = ""
        for doc in documents:
            doc_str += doc.text
        processed_nodes.append(NodeWithScore(node=Node(text=doc_str), score=1.0))

    # Retrieve the response from LLM
    response = response_synthesizer.synthesize(
        query=user_query,
        nodes=processed_nodes,
        additional_context={"context_str": "context_str", "query_str": user_query},
    )

    # Print LLM template for debugging
    messages = llm_template_logger.get_messages()
    for message in messages:
        print(f"<DEBUG> {message.content}")
    return response.response


while True:
    user_input = input("Enter your query: ")
    answer = answer_query(user_input)
    print("Answer:", answer)
