from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.core.indices import VectorStoreIndex
from llama_index.multi_modal_llms.replicate.base import REPLICATE_MULTI_MODAL_LLM_MODELS
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import csv
import os
import pandas as pd
from tqdm import tqdm
from metrics import eval_dimensions_qa


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
    csv_name = f"dimension_{question_type}_evaluation_{model}.csv"
    if not os.path.exists(csv_name) or overwrite_answers:
        questions_pd = pd.read_csv(f"../../dataset/rule_compliance/rule_dimension_qa/{question_type}/rule_dimension_qa_{question_type}.csv")
        questions_pd.to_csv(csv_name, index=False)
    else:
        questions_pd = pd.read_csv(csv_name)
    return questions_pd, csv_name


def run_thread(model, question, image_path, context):
    if model == 'llava-13b':
        # API token of the model/pipeline that we will be using
        REPLICATE_API_TOKEN = ""
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        model = REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"]
        multi_modal_llm = ReplicateMultiModal(model=model, max_new_tokens=100)
    elif model in ['gpt-4-1106-vision-preview', 'gpt-4-1106-vision-preview+RAG']:
        # OpenAI model
        multi_modal_llm = OpenAIMultiModal(
            model='gpt-4-vision-preview', max_new_tokens=1500
        )
    else:
        raise ValueError("Invalid model")

    # load question image
    image_document = SimpleDirectoryReader(input_files=[image_path]).load_data()

    # modify text prompt to include context
    question = add_context_to_prompt(question, context)

    # get response from model
    rag_response = multi_modal_llm.complete(prompt=question, image_documents=image_document)
    return str(rag_response)


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

    index.storage_context.persist(f"index-{chunk_size}")
    return index


def retrieve_context(index, question, top_k=10):
    if top_k == 0:
        # load all context from original text document
        txt_path = "../../dataset/docs/rules_pdfplumber1.txt"
        context = open(txt_path, "r", encoding="utf-8").read()
    else:
        retriever = index.as_retriever(similarity_top_k=top_k)
        context = retriever.retrieve(question)
    return context


def save_results(model, macro_avg_accuracy, direct_dim_avg, scale_bar_avg, all_accuracies, macro_avg_bleus,
                         all_bleus, macro_avg_rogues, all_rogues):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nDirect Dimension avg: {direct_dim_avg}")
    print(f"\nScale Bar avg: {scale_bar_avg}")
    print(f"\nAll accuracies: {all_accuracies}")
    print(f"\nMacro avg bleus: {macro_avg_bleus}")
    print(f"\nAll bleus: {all_bleus}")
    print(f"\nMacro avg rogues: {macro_avg_rogues}")
    print(f"\nAll rogues: {all_rogues}")

    # Save results to txt file
    with open(f"dimension_{question_type}_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nDirect Dimension avg: {direct_dim_avg}")
        text_file.write(f"\nScale Bar avg: {scale_bar_avg}")
        text_file.write(f"\nAll accuracies: {all_accuracies}")
        text_file.write(f"\nMacro avg bleus: {macro_avg_bleus}")
        text_file.write(f"\nAll bleus: {all_bleus}")
        text_file.write(f"\nMacro avg rogues: {macro_avg_rogues}")
        text_file.write(f"\nAll rogues: {all_rogues}")


if __name__ == '__main__':
    overwrite_answers = False

    # Index the text data
    if os.path.exists("index"):
        print("Loading index from storage...")
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="index")
        # load index
        index = load_index_from_storage(storage_context, embed_model=OpenAIEmbedding(model='text-embedding-3-large'))
    else:
        print("Creating index...")
        index = create_index()
        index.storage_context.persist("index")

    for question_type in ["context", "detailed_context"]:
        for model in ['gpt-4-1106-vision-preview', 'gpt-4-1106-vision-preview+RAG', 'llava-13b']:
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
                image_path = f"../../dataset/rule_compliance/rule_dimension_qa/{question_type}/" + row['image']

                # Run through model
                if model in ['gpt-4-1106-vision-preview+RAG', 'llava-13b']:
                    context = retrieve_context(index, question, top_k=12)
                elif model in ['gpt-4-1106-vision-preview']:
                    context = retrieve_context(index, question, top_k=0)
                else:
                    raise ValueError("Invalid model")

                try:
                    response = run_thread(model, question, image_path, context)
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
            macro_avg_accuracy, direct_dim_avg, scale_bar_avg, all_accuracies, macro_avg_bleus, all_bleus, \
                macro_avg_rogues, all_rogues = eval_dimensions_qa(csv_name)

            # Print and save the results
            save_results(model, macro_avg_accuracy, direct_dim_avg, scale_bar_avg, all_accuracies, macro_avg_bleus,
                         all_bleus, macro_avg_rogues, all_rogues)
