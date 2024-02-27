from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.replicate.base import (
    REPLICATE_MULTI_MODAL_LLM_MODELS,
)
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import qdrant_client
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


if __name__ == '__main__':

    # model = 'gpt-4-vision-preview'
    model = 'llava-13b'

    if model == 'llava-13b':
        top_k = 1
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
    elif model == 'gpt-4-vision-preview':
        top_k = 1
        # OpenAI model
        multi_modal_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview", max_new_tokens=1500
        )
    else:
        raise ValueError("Invalid model")

    # Collect all text query
    query_path = 'rule_definition_qa.csv'
    queries = get_text_prompts(query_path)
    query_str = queries[1]

    # Collect all image query
    image_path = './rule_definition_qa/1'
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
        documents + image_documents,
        storage_context=storage_context,
        embed_model='local'
    )

    query_engine = index.as_query_engine(llm=multi_modal_llm, similarity_top_k=top_k, verbose=True)

    rag_response = query_engine.query(query_str)

    print(query_str)
    print(str(rag_response))

    # display the source
    print("-text sources:")
    for item in rag_response.metadata["text_nodes"]:
        print(item.metadata['page_label'])
        print(item.score)
    print("\n-image sources:")
    for item in rag_response.metadata["image_nodes"]:
        print(item.metadata['file_path'])
        print(item.score)
        print()
