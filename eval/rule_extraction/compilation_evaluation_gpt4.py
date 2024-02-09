import os
import re
import string
from collections import Counter
from openai import OpenAI
from time import sleep
import pandas as pd
from tqdm import tqdm
import ast


def upload_file(file_path):
    # Upload a file with an "assistants" purpose
    upload_file = client.files.create(
        file=open(file_path, "rb"),
        purpose='assistants')

    return upload_file


def create_assistant(file):
    # Add the file to the assistant
    assistant = client.beta.assistants.create(
        instructions="You are a mechanical designer chatbot. Use your knowledge base to best respond to customer queries.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file.id])

    return assistant


def run_thread(question):
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id, )
    status_complete = False
    i = 0
    while status_complete is False and i < 0.5*1000:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id)
        status_complete = run.status == "completed"
        sleep(0.5)
        i += 1

        # Retrieve the message object
        messages = client.beta.threads.messages.list(
            thread_id=thread.id)

    return messages.data[0].content[0].text.value


if __name__ == '__main__':
    # get the api key from memory
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    from config import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)

    file_path = '../../dataset/rule_extraction/docs/FSAE_Rules_2024_V1.pdf'
    file = upload_file(file_path)
    assistant = create_assistant(file)

    questions_pd = pd.read_csv("../../dataset/rule_extraction/qa_compilation/data/compilation.csv")

    # Get responses from the model and compute accuracy
    questions_pd['response'] = None
    for index, row in tqdm(questions_pd.iterrows(), total=len(questions_pd)):
        question = row['question']

        # Run through model
        response = run_thread(question)

        # Save the response
        response = response.split("【")[0].split(', ')
        questions_pd.at[index, 'response'] = response

        # compute the accuracy of the response:
        ground_truth = ast.literal_eval(row['answer'])
        score = 0   # TODO
        questions_pd.at[index, 'f1_score'] = score


    # save the results
    questions_pd.to_csv("compilation_evaluation_gpt4.csv", index=False)

    # print the average of the scores
    print(f"Average F1 Score: {questions_pd['f1_score'].mean()}")
