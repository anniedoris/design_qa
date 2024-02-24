from llama_index.core import ( 
    Settings,
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    PromptTemplate, 
    )
from llama_index.core.schema import ImageDocument   
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.multi_modal_llms.replicate.base import (
    REPLICATE_MULTI_MODAL_LLM_MODELS,
)
import qdrant_client
from pathlib import Path
import csv
import os 


def get_text_prompts(text_query_path):
    # get prompt dataset
    # text prompt 
    queries = []
    with open(text_query_path, mode='r') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        for row in csv_reader:
            queries.append(row[0])
    return queries


# API token of the model/pipeline that we will be using 
REPLICATE_API_TOKEN = "r8_HMUyttnOdSTWaPEKc6irVMcK1y3pIkS2jBoly"
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Replicate support: llava-13b, fuyu-8b, minigpt-4, cogvlm
model = REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"]
multi_modal_llm = ReplicateMultiModal(
                model=model,
                max_new_tokens=100,
                temperature=0.1,
                num_input_files=1,
                top_p=0.9,
                num_beams=1,
                repetition_penalty=1,
            )


# Collect all text query
query_path = 'rule_definition_qa.csv'  
queries = get_text_prompts(query_path)
query_str = queries[1]

# Collect all image query
image_path = './rule_definition_qa/1/'
image_documents = SimpleDirectoryReader(image_path).load_data()

# step 1: Collect all pdf data 
pdf_path = './data/'
documents = SimpleDirectoryReader(pdf_path).load_data()


# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_mm_db")
text_store = QdrantVectorStore(client=client, collection_name="text_collection")
image_store = QdrantVectorStore(client=client, collection_name="image_collection")
storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)


# create the multimodal vector index
index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model='local'
    )


# --------------------------------------------------
# how to include the image into the query?
# --------------------------------------------------
imageUrl = 'rule_definition_qa/1/1.jpg'
llava_response = multi_modal_llm.complete(
    prompt=query_str,
    image_documents=[ImageDocument(image_path=imageUrl)],
)


query_engine = index.as_query_engine(llm=multi_modal_llm, similarity_top_k=2, verbose=True)

prompt_template = "please provide relevant information about: "
rag_response = query_engine.query(prompt_template + llava_response.text)

print(str(rag_response))

