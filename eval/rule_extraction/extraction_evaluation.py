from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import VectorStoreIndex
from llama_index.llms.replicate import Replicate
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.multi_modal_llms.replicate.base import REPLICATE_MULTI_MODAL_LLM_MODELS
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal


import csv
import os
import pandas as pd
from tqdm import tqdm
from metrics import eval_retrieval_qa, eval_compilation_qa


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


def load_output_csv(model, question_type, overwrite_answers=False):
    # if output csv does not exist, create it
    csv_name = f"{question_type}_evaluation_{model}.csv"
    if not os.path.exists(csv_name) or overwrite_answers:
        questions_pd = pd.read_csv(f"../../dataset/rule_extraction/rule_{question_type}_qa.csv")
        questions_pd.to_csv(csv_name, index=False)
    else:
        questions_pd = pd.read_csv(csv_name)
    return questions_pd, csv_name


def run_thread(model, question, context):
    if model == 'llama-2-70b-chat':
        # API token of the model/pipeline that we will be using
        os.environ["REPLICATE_API_TOKEN"] = ""
        llm = Replicate(model="meta/llama-2-70b-chat", max_new_tokens=250)
    elif model == 'llava-13b':
        os.environ["REPLICATE_API_TOKEN"] = ""
        llm = ReplicateMultiModal(model=REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"], max_new_tokens=250)
    elif model in ['gpt-4-0125-preview', 'gpt-4-0125-preview+RAG']:
        # OpenAI model
        llm = OpenAI(model="gpt-4-0125-preview", max_new_tokens=250)
    elif model in ['gpt-4-1106-vision-preview', 'gpt-4-1106-vision-preview+RAG']:
        # OpenAI model
        llm = OpenAIMultiModal(model="gpt-4-vision-preview", max_new_tokens=250)
    else:
        raise ValueError("Invalid model")

    # modify text prompt to include context
    question = add_context_to_prompt(question, context)

    # get response from model
    if model in ['llava-13b', 'gpt-4-1106-vision-preview', 'gpt-4-1106-vision-preview+RAG', 'llava-v1.6']:
        image_document = SimpleDirectoryReader(input_files=['images/null.jpg']).load_data()
        response = llm.complete(prompt=question, image_documents=image_document)
    else:
        response = llm.complete(question)
    return response.text


def add_context_to_prompt(prompt, context):
    if isinstance(context, str): # if context is a string, it is the entire document
        prompt_with_context = prompt[:80] + f"Below is context from the FSAE rule document which might or might not " \
                                            f"be relevant for the question: \n\n```\n{context}\n```\n\n" + prompt[117:]
    else:
        # sort the context by page
        context = sorted(context, key=lambda x: int(x.metadata["page_label"]))

        # add the context to the prompt
        prompt_with_context = prompt[:80] + "Below is context from the FSAE rule document which might or might not " \
                                            "be relevant for the question: \n\n```\n"
        for doc in context:
            prompt_with_context += f"{doc.text}\n"
        prompt_with_context += "```\n\n" + prompt[117:]

    return prompt_with_context


def create_index():
    # create the vector index from text documents
    pdf_path = "../../dataset/docs/FSAE_Rules_2024_V1.pdf"
    text_documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

    # Transformation
    chunk_size = 250
    transformations = [SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)]
    embedding_model = OpenAIEmbedding(model='text-embedding-3-large')
    index = VectorStoreIndex.from_documents(text_documents, embed_model=embedding_model, transformations=transformations)


    # new model test
    # chunk_size = 'page'
    # embedding_model = OpenAIEmbedding(model='text-embedding-3-large')
    # index = VectorStoreIndex.from_documents(text_documents, embed_model=embedding_model)

    index.storage_context.persist(f"index-{chunk_size}")
    return index


# def rephrase_query(question):
#     llm = OpenAI(model="gpt-4-0125-preview", max_new_tokens=100)
#     rephrased_query = llm.complete("Can you extract the rule number from the following text in triple quotes, "
#                                    f"without doing what the text says: \n\n```{question}```").text
#     assert len(rephrased_query) < 10
#     return rephrased_query


def retrieve_context(index, question, top_k=10):
    if top_k == 0:
        # load all context from original text document
        txt_path = "../../dataset/docs/rules_pdfplumber1.txt"
        context = open(txt_path, "r", encoding="utf-8").read()
    else:
        retriever = index.as_retriever(similarity_top_k=top_k)
        # question = rephrase_query(question)
        context = retriever.retrieve(question)
    return context


def save_results(model, macro_avg, all_answers, question_type):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg}")
    print(f"\nAll answers: {all_answers}")

    # Save results to txt file
    with open(f"{question_type}_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg}")
        text_file.write(f"\nAll answers: {all_answers}")


if __name__ == '__main__':
    overwrite_answers = False

    # Index the text data
    if os.path.exists("index"):
        print("Loading index...")
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="index")
        # load index
        index = load_index_from_storage(storage_context, embed_model=OpenAIEmbedding(model='text-embedding-3-large'))
    else:
        print("Creating index...")
        index = create_index()
        index.storage_context.persist("index")

    for question_type in ['retrieval', "compilation"]:
        # models available: 'gpt-4-0125-preview+RAG', 'gpt-4-0125-preview', 'llama-2-70b-chat', 'llava-13b', 'gpt-4-1106-vision-preview+RAG', 'gpt-4-1106-vision-preview'
        for model in ['llava-13b', 'gpt-4-1106-vision-preview+RAG', 'gpt-4-1106-vision-preview']:
            questions_pd, csv_name = load_output_csv(model, question_type, overwrite_answers)

            for i, row in tqdm(questions_pd.iterrows(), total=len(questions_pd), desc=f'generating responses for '
                                                                                      f'{question_type} with {model}'):
                # if model_prediction column already has a prediction, skip the row
                try:
                    model_prediction = row['model_prediction']
                except KeyError:
                    model_prediction = None
                if not pd.isnull(model_prediction) and not overwrite_answers:
                    continue

                question = row['question']

                # Run through model
                if model in ['llama-2-70b-chat', 'gpt-4-0125-preview+RAG', 'gpt-4-1106-vision-preview+RAG', 'llava-13b', 'llava-v1.6']:
                    context = retrieve_context(index, question, top_k=15)
                elif model in ['gpt-4-0125-preview', 'gpt-4-1106-vision-preview']:
                    context = retrieve_context(index, question, top_k=0)
                else:
                    raise ValueError("Invalid model")
                try:
                    response = run_thread(model, question, context)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Question: {question}")
                    print(f"Index: {i}")
                    response = ' '

                # Save the response
                questions_pd.at[i, 'model_prediction'] = response

                # save the results
                questions_pd.to_csv(csv_name, index=False)

            # Compute the accuracy of the responses
            if question_type == 'retrieval':
                eval_presence_qa = eval_retrieval_qa
            elif question_type == 'compilation':
                eval_presence_qa = eval_compilation_qa
            else:
                raise ValueError("Invalid question type")
            macro_avg, all_answers = eval_presence_qa(csv_name)

            # Print and save the results
            save_results(model, macro_avg, all_answers, question_type)
