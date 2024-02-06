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


def combine_prediction(row):
    rule_number = row['rule_number']
    rule_text = row['rule_text']
    rule_title = row['rule_title']

    if rule_number is None or pd.isnull(rule_number):
        rule_number = ""
    if rule_text is None or pd.isnull(rule_text):
        rule_text = ""
    if rule_title is None or pd.isnull(rule_title):
        rule_title = ""

    return ' '.join([rule_number, rule_title, rule_text])


if __name__ == '__main__':
    # get the api key from memory
    from config import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)

    file_path = 'docs/FSAE_Rules_2024_V1 (2).pdf'
    file = upload_file(file_path)
    assistant = create_assistant(file)

    questions_pd = pd.read_csv("docs/FSAE DesignSpecQA Benchmark - retrieval.csv")
    for index, row in tqdm(questions_pd.iterrows(), desc='generating responses', total=len(questions_pd)):
        # if response column is not empty, skip the row
        try:
            response = row['response']
        except KeyError:
            response = None
        if not pd.isnull(response):
            continue

        rule_number = row['rule_number']
        rule_text = row['rule_text']

        # Exclude the following questions
        if rule_text is None or pd.isnull(rule_text) or rule_number.split('.')[0] in ['GR', 'AR', 'DR']:
            continue

        # Compile the question
        question = f"We are a student engineering team designing a vehicle for the FSAE competition. Attached is the " \
                   f"FSAE rules document. What does rule {rule_number} state exactly? Answer with only the text of " \
                   f"the rule and no other words."

        # Run through model
        response = run_thread(question)

        # Save the response
        questions_pd.loc[index, 'question'] = question
        questions_pd.loc[index, 'response'] = response.split("【")[0] + "."
        questions_pd.to_csv("docs/FSAE DesignSpecQA Benchmark - retrieval.csv", index=False)

    # compute the f1 score between the answer and the response
    questions_pd['f1 score'] = questions_pd[questions_pd[['rule_text', 'response']].notnull().all(1)].apply(lambda row: token_f1_score(combine_prediction(row), row['response']), axis=1)

    # save the results
    questions_pd.to_csv("docs/FSAE DesignSpecQA Benchmark - retrieval.csv", index=False)

    # print the average of the f1 scores
    print(f"Average F1 Score: {questions_pd['f1 score'].mean()}")