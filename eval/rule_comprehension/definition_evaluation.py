from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    )
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.replicate.base import REPLICATE_MULTI_MODAL_LLM_MODELS
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import qdrant_client
import csv
import os
import pandas as pd
from tqdm import tqdm
from metrics import eval_definition_qa


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


def load_output_csv(model):
    # if output csv does not exist, create it
    csv_name = f"definition_evaluation_{model}.csv"
    if not os.path.exists(csv_name):
        questions_pd = pd.read_csv("../../dataset/rule_comprehension/rule_definition_qa.csv")
        questions_pd.to_csv(csv_name, index=False)
    else:
        questions_pd = pd.read_csv(csv_name)
    return questions_pd, csv_name


def run_thread(model, question, image_path):
    if model == 'llava-13b':
        top_k = 1
        # API token of the model/pipeline that we will be using
        REPLICATE_API_TOKEN = ""
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
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
        top_k = 5
        # OpenAI model
        multi_modal_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview", max_new_tokens=1500
        )
    else:
        raise ValueError("Invalid model")

    image_documents = SimpleDirectoryReader(input_files=[image_path]).load_data()
    pdf_path = "../../dataset/docs/FSAE_Rules_2024_V1.pdf"
    text_documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path="qdrant_mm_db")
    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    client.delete_collection(collection_name="image_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")
    storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

    # create the multimodal vector index
    index = MultiModalVectorStoreIndex.from_documents(
        text_documents + image_documents,
        storage_context=storage_context,
        embed_model='local'
    )

    query_engine = index.as_query_engine(llm=multi_modal_llm, similarity_top_k=top_k, verbose=True)

    rag_response = query_engine.query(question)

    print(question)
    print(str(rag_response))

    # display the source
    print("\n-text sources:")
    for item in rag_response.metadata["text_nodes"]:
        print(item.metadata['page_label'])
        # print(item.score)
    print("\n-image sources:")
    for item in rag_response.metadata["image_nodes"]:
        print(item.metadata['file_path'])
        # print(str(item.score) + '\n')

    return str(rag_response)


if __name__ == '__main__':
    for model in ['llava-13b', 'gpt-4-vision-preview']:
        questions_pd, csv_name = load_output_csv(model)

        for index, row in tqdm(questions_pd.iterrows(), total=len(questions_pd), desc=f'generating responses for {model}'):
            # if model_prediction column already has a prediction, skip the row
            try:
                model_prediction = row['model_prediction']
            except KeyError:
                model_prediction = None
            if not pd.isnull(model_prediction):
                continue

            question = row['question']
            image_path = "../../dataset/rule_comprehension/rule_definition_qa/" + row['image']

            # Run through model
            response = run_thread(model, question[:322], image_path)

            # Save the response
            questions_pd.at[index, 'model_prediction'] = response

            # save the results
            questions_pd.to_csv(csv_name, index=False)

        # Compute the accuracy of the responses
        macro_avg, definitions_avg, multi_avg, single_avg, all_answers = eval_definition_qa(csv_name)

        # Print the results
        print(f"\nMacro avg: {macro_avg}")
        print(f"\nDefinitions: {definitions_avg}")
        print(f"\nMulti avg: {multi_avg}")
        print(f"\nSingle avg: {single_avg}")
        print(f"\nAll answers: {all_answers}")


