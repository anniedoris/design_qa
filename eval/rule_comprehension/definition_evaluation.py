from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.core.indices import VectorStoreIndex
from llama_index.multi_modal_llms.replicate.base import REPLICATE_MULTI_MODAL_LLM_MODELS
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
import os
import pandas as pd
from tqdm import tqdm
from metrics import eval_definition_qa


def load_output_csv(model, overwrite_answers=False):
    # if output csv does not exist, create it
    csv_name = f"definition_evaluation_{model}.csv"
    if not os.path.exists(csv_name) or overwrite_answers:
        questions_pd = pd.read_csv("../../dataset/rule_comprehension/rule_definition_qa.csv")
        questions_pd.to_csv(csv_name, index=False)
    else:
        questions_pd = pd.read_csv(csv_name)
    return questions_pd, csv_name


def run_thread(model, question, image_path):
    if model == 'llava-13b':
        # API token of the model/pipeline that we will be using
        REPLICATE_API_TOKEN = ""
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
        model = REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"]
        multi_modal_llm = ReplicateMultiModal(model=model, max_new_tokens=100)
    elif model == 'gpt-4-1106-vision-preview' or model == 'gpt-4-1106-vision-preview+context':
        # OpenAI model
        multi_modal_llm = OpenAIMultiModal(model="gpt-4-vision-preview", max_new_tokens=100)
    else:
        raise ValueError("Invalid model")

    # load question image
    image_document = SimpleDirectoryReader(input_files=[image_path]).load_data()

    # get response from model
    rag_response = multi_modal_llm.complete(prompt=question, image_documents=image_document)
    return str(rag_response)


def save_results(model, macro_avg, definitions_avg, multi_avg, single_avg, all_answers):
    print(f"\nMacro avg: {macro_avg}")
    print(f"\nDefinitions: {definitions_avg}")
    print(f"\nMulti avg: {multi_avg}")
    print(f"\nSingle avg: {single_avg}")
    print(f"\nAll answers: {all_answers}")

    # Save results to txt file
    with open(f"definition_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg}")
        text_file.write(f"\nDefinitions: {definitions_avg}")
        text_file.write(f"\nMulti avg: {multi_avg}")
        text_file.write(f"\nSingle avg: {single_avg}")
        text_file.write(f"\nAll answers: {all_answers}")


def retrieve_context(question):
    # load all context from original text document
    txt_path = "../../dataset/docs/rules_pdfplumber1.txt"
    context = open(txt_path, "r", encoding="utf-8").read()

    question_with_context = question[:80] + f"Below is context from the FSAE rule document which might or might not " \
                            f"be relevant for the question: \n\n```\n{context}\n```\n\n" + question[117:]

    return question_with_context


if __name__ == '__main__':
    overwrite_answers = False

    for model in ['gpt-4-1106-vision-preview+context', 'gpt-4-1106-vision-preview', 'llava-13b']:
        questions_pd, csv_name = load_output_csv(model, overwrite_answers=overwrite_answers)

        for i, row in tqdm(questions_pd.iterrows(), total=len(questions_pd), desc=f'generating responses for {model}'):
            # if model_prediction column already has a prediction, skip the row
            try:
                model_prediction = row['model_prediction']
            except KeyError:
                model_prediction = None
            if not pd.isnull(model_prediction) and not overwrite_answers:
                continue

            question = row['question']
            image_path = "../../dataset/rule_comprehension/rule_definition_qa/" + row['image']

            # Run through model
            if model == 'gpt-4-1106-vision-preview+context':
                question = retrieve_context(question)
            response = run_thread(model, question, image_path)

            # Save the response
            questions_pd.at[i, 'model_prediction'] = response

            # save the results
            questions_pd.to_csv(csv_name, index=False)

        # Compute the accuracy of the responses
        macro_avg, definitions_avg, multi_avg, single_avg, all_answers = eval_definition_qa(csv_name)

        # Print and save the results
        save_results(model, macro_avg, definitions_avg, multi_avg, single_avg, all_answers)