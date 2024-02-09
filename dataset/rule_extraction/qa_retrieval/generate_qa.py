import re
import string
from collections import Counter
from openai import OpenAI
from time import sleep
import pandas as pd
from tqdm import tqdm

def compile_answer(row):
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
    questions_pd = pd.read_csv("../rules_pdfplumber1_clean1.csv", encoding='utf-8-sig')
    qa = []
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

        qa.append([question, compile_answer(row)])

    # Export questions and answers to compilation.csv
    pd.DataFrame(qa, columns=['question', 'answer']).to_csv("data/rule_retrieval_qa.csv", index=False)

    print(len(qa))
