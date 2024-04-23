from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.replicate import ReplicateMultiModal
from llama_index.core.indices import VectorStoreIndex
from llama_index.multi_modal_llms.replicate.base import REPLICATE_MULTI_MODAL_LLM_MODELS
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
import os
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("../metrics/")
sys.path.append("../")
from metrics import eval_definition_qa
from model_list import model_list
from PIL import Image


def load_output_csv(model, overwrite_answers=False):
    # if output csv does not exist, create it
    csv_name = f"definition_evaluation_{model}.csv"
    if not os.path.exists(csv_name) or overwrite_answers:
        questions_pd = pd.read_csv("../../dataset/rule_comprehension/rule_definition_qa.csv")
        questions_pd.to_csv(csv_name, index=False)
    else:
        questions_pd = pd.read_csv(csv_name)
    return questions_pd, csv_name

def convert_and_optimize_jpg_to_png(jpg_file_path, png_file_path, max_size_mb=3.3):
    # Load the image
    with Image.open(jpg_file_path) as img:
        # Convert and save initially to check size
        img.save(png_file_path, 'PNG')

        # Check file size and reduce resolution if necessary
        file_size = os.path.getsize(png_file_path)
        max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes

        while file_size > max_size_bytes:
            # Reduce both dimensions by 10%
            width, height = img.size
            img = img.resize((int(width * 0.9), int(height * 0.9)), Image.LANCZOS)

            # Save and check again
            img.save(png_file_path, 'PNG', optimize=True)
            file_size = os.path.getsize(png_file_path)

def run_thread(model, question, image_path):
    if model == 'llava-13b':
        # API token of the model/pipeline that we will be using
        model = REPLICATE_MULTI_MODAL_LLM_MODELS["llava-13b"]
        multi_modal_llm = ReplicateMultiModal(model=model, max_new_tokens=100)
    elif model == 'gpt-4-1106-vision-preview' or model == 'gpt-4-1106-vision-preview+context':
        # OpenAI model
        multi_modal_llm = OpenAIMultiModal(model="gpt-4-vision-preview", max_new_tokens=100)
    elif model in ['gemini-pro']:
        multi_modal_llm = GeminiMultiModal(model_name='models/gemini-pro-vision')
    elif model in ['claude-opus-RAG']:
        multi_modal_llm = AnthropicMultiModal(model="claude-3-opus-20240229")
        print("Converting image to png")
        convert_and_optimize_jpg_to_png(image_path, image_path.strip('jpg') + 'png')
        print("Finished converting image to png")
        image_path = image_path.strip('jpg') + 'png'
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
    overwrite_answers = True

    for model in model_list:
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