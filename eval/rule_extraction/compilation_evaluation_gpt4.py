from openai import OpenAI
from time import sleep
import pandas as pd
from tqdm import tqdm
import os
from metrics import eval_compilation_qa

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
    client = OpenAI()

    file_path = '../../dataset/docs/FSAE_Rules_2024_V1.pdf'
    file = upload_file(file_path)
    assistant = create_assistant(file)

    questions_pd = pd.read_csv("../../dataset/rule_extraction/rule_compilation_qa.csv")

    # if output csv does not exist, create it
    csv_name = "compilation_evaluation_gpt4.csv"
    if not os.path.exists(csv_name):
        questions_pd.to_csv(csv_name, index=False)
    else:
        questions_pd = pd.read_csv(csv_name)

    # Get responses from the model
    for index, row in tqdm(questions_pd.iterrows(), total=len(questions_pd), desc='generating responses'):
        # if model_prediction column already has a prediction, skip the row
        try:
            model_prediction = row['model_prediction']
        except KeyError:
            model_prediction = None
        if not pd.isnull(model_prediction):
            continue

        question = row['question']

        # Run through model
        response = run_thread(question)

        # Save the response
        response = response.split("【")[0].split(', ')
        questions_pd.at[index, 'model_prediction'] = response

    # save the results
    questions_pd.to_csv(csv_name, index=False)

    # Compute the accuracy of the responses
    macro_avg, all_answers = eval_compilation_qa(csv_name)

    # print the average of the scores
    print(f"Average F1 Score: {macro_avg}")
    print(f"\nIndividual F1 Scores: {all_answers}")
