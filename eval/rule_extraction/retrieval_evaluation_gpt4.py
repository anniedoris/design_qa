import os
import re
import string
from collections import Counter
from openai import OpenAI
from time import sleep
import pandas as pd
from tqdm import tqdm


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


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


if __name__ == '__main__':
    # get the api key from memory
    client = OpenAI()

    file_path = '../../dataset/docs/FSAE_Rules_2024_V1.pdf'
    file = upload_file(file_path)
    assistant = create_assistant(file)

    questions_pd = pd.read_csv("../../dataset/rule_extraction/rule_retrieval_qa.csv")
    # if output csv does not exist, create it
    if not os.path.exists("retrieval_evaluation_gpt4.csv"):
        questions_pd.to_csv("retrieval_evaluation_gpt4.csv", index=False)
    else:
        questions_pd = pd.read_csv("retrieval_evaluation_gpt4.csv")

    for index, row in tqdm(questions_pd.iterrows(), desc='generating responses', total=len(questions_pd)):
        # if response column already has a response, skip the row
        try:
            response = row['response']
        except KeyError:
            response = None
        if not pd.isnull(response):
            continue

        question = row['question']
        answer = row['answer']

        # Run through model
        response = run_thread(question).split("【")[0] + "."

        # Save the response and calculate f1 score
        questions_pd.loc[index, 'response'] = response
        questions_pd.loc[index, 'f1 score'] = token_f1_score(answer, response)

        # save to disk after each iteration
        questions_pd.to_csv("retrieval_evaluation_gpt4.csv", index=False)

    # print the average of the f1 scores
    print(f"Average F1 Score: {questions_pd['f1 score'].mean()}")
