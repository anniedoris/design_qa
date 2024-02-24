import re
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import pandas as pd
from statistics import mean
import textdistance

########################
## Cleaning functions
########################
def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    # This replaces any newline characters or tab characters with a space
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def character_string_no_space(text):
    return text.replace(' ', '')

def clean_rule_list_prediction(rl):
    list_of_rules = rl.split(',')
    list_of_rules = [i.strip() for i in list_of_rules]
    return " ".join(list_of_rules)

# TODO: fix compilation.csv ground truth so we don't have to handle GT differently from prediction
def clean_rule_list_ground_truth(rl):
    rl = rl.strip('[')
    rl = rl.strip(']')
    list_of_rules = rl.split(',')
    list_of_rules = [i.strip() for i in list_of_rules]
    list_of_rules = [i.strip("'") for i in list_of_rules]
    return " ".join(list_of_rules)

########################
## F1 functions
########################

# # F1 on a bag of word basis
# def token_f1_score(prediction, ground_truth):
#     """
#     Taken from the official evaluation script for v1.1 of the SQuAD dataset.
#     """
#     prediction_tokens = normalize_answer(prediction).split()
#     ground_truth_tokens = normalize_answer(ground_truth).split()
#     # Counts the number of times there's a match between the prediction and the ground truth
#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(ground_truth_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1

# F1 on a bag of tokens
def token_f1_score(prediction_tokens, ground_truth_tokens):
    """
    Based on the official evaluation script for v1.1 of the SQuAD dataset.
    """
    # Counts the number of times there's a match between the prediction and the ground truth
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

########################
## BLEU
########################
def tokenize(text):
    """
    Taken from the ScienceQA evaluations.py script
    """
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens

def bleu_score(target, prediction, gram):
    """
    Taken from the ScienceQA evaluations.py script
    """

    reference_tokens = tokenize(target)
    hypothesis_tokens = tokenize(prediction)

    print("Reference tokens")
    print(reference_tokens)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu

########################
## Rouge-L
########################
def score_rouge(target, prediction):
    """
    Taken from the ScienceQA evaluations.py script
    """
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(target, prediction, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l