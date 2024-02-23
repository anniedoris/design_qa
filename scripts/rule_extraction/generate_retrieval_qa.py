import re
import string
from collections import Counter
from time import sleep
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append("..")
from common_prompts import prompt_preamble


def compile_answer(row):
    rule_text = row['rule_text']

    # Filter out rules that are less than 40 characters, these are likely just title rules
    if len(rule_text) < 40 or rule_text is None:
        return None
    else:
        return rule_text


if __name__ == '__main__':
    questions_pd = pd.read_csv("../../dataset/docs/csv_rules/all_rules_extracted.csv", encoding='utf-8-sig')

    qa = []
    for index, row in tqdm(questions_pd.iterrows(), desc='generating responses', total=len(questions_pd)):
        # if response column is not empty, skip the row
        try:
            response = row['response']
        except KeyError:
            response = None
        if not pd.isnull(response):
            continue

        rule_number = row['rule_num']
        rule_text = row['rule_text']

        # Exclude the following questions
        if pd.isnull(rule_text) or rule_number.split('.')[0] in ['GR', 'AR', 'DR']:
            continue

        # Compile the question
        question = prompt_preamble + f"What does rule {rule_number} state exactly? Answer with only the text of " \
                                     f"the rule and no other words."

        answer = compile_answer(row)

        if answer != None:
            qa.append([question, answer])

    # Export questions and answers to compilation.csv
    pd.DataFrame(qa, columns=['question', 'answer']).to_csv("../../dataset/rule_extraction/rule_retrieval_qa.csv",
                                                            index=False)

    print(len(qa))
