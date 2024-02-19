
import re
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import pandas as pd
from statistics import mean

# METRICS IN LITERATURE
# Macro-averaged F1, use bags of tokens to compute accuracy
    ## Used by SQuAD dataset, answer is correct span (verbatim) from a wikipedia passage (model chooses which span)
    ## QASPER uses the same metric as SQUAD
    ## F1, precision, and recall used by WikiQA, which contains answers that are verbatim sentences from a wikipedia passage
# BLEU/ROGUE
    ## Used by ScienceQA for scoring of the text generation prompts (specifically, they use BLEU-1, BLEU-4, and ROGUE-L)
    ## Note that Bleu performs better when there are more reference samples
    ## Nougat also uses BLEU
    ## Some tasks from ZeroSrolls use ROGUE, interestingly they use geometric mean of ROGUE-1, ROGUE-2, and ROGUE-L
    ## Found sources that explains BLEU is precision-focused while ROGUE is recall-focsued
# Sentence similarity
    ## Used by ScienceQA (in addition to BLUE and ROGUE)
    ## Computes the cosine-similarity of semantic embeddings between two sentences

# OUR TASKS
# Rule Extraction
    ## Retrieval - F1 score?
    ## Compilation - F1 score
# Rule Comprehension
    ## Definition - F1 score
    ## Presence - F1 score
# Rule Evaluation
    ## Dimensioning & tolerancing
    ## Functional performance

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

########################
## Evals per QA dataset
########################
# Retrieval QAs will be scored using F1 on bag of words
def eval_retrieval_qa(results_csv):
    """
    :param results_csv: the csv should contain the results from running the QA through a model.
    it should have a column called "model_prediction" and another called "ground_truth" with corresponding GT
    
    :returns: 
    1. overall score on QA (macro average)
    2. F1 score for each QA
    """
    results_df = pd.read_csv(results_csv)
    f1_scores = []
    for i, row in results_df.iterrows():
        prediction_tokens = normalize_answer(row['model_prediction']).split()
        ground_truth_tokens = normalize_answer(row['ground_truth']).split()
        f1_scores.append(token_f1_score(prediction_tokens, ground_truth_tokens))
        
    return mean(f1_scores), f1_scores

# Compilation QAs will be scored using F1 on rule numbers
def eval_compilation_qa(results_csv):
    """
    :param results_csv: the csv should contain the results from running the QA through a model.
    it should have a column called "model_prediction" and another called "ground_truth" with corresponding GT
    
    :returns: 
    1. overall score on QA (macro average)
    2. F1 score for each QA
    """
    results_df = pd.read_csv(results_csv)
    f1_scores = []
    for i, row in results_df.iterrows():
        prediction_tokens = clean_rule_list_prediction(row['model_prediction']).split()
        ground_truth_tokens = clean_rule_list_ground_truth(row['ground_truth']).split()
        f1_scores.append(token_f1_score(prediction_tokens, ground_truth_tokens))
        
    return mean(f1_scores), f1_scores

# Definition QAs will be scored using max F1 across all synonyms
# TODO: think about whether this is the best metric for this category. I think it makes sense as long as all synonyms for
# the component(s) are directly mentioned in the document
def eval_definition_qa(prediction, ground_truth):
    # ground_truth_list contains a list of possible synonym answers to the question, separated by commas
    synonyms_ground_truth = ground_truth.split(',')
    synonyms_ground_truth = [i.strip(' ') for i in synonyms_ground_truth]
    f1_scores = []
    for synonym in synonyms_ground_truth:
        f1_scores.append(token_f1_score(prediction, synonym))

    # Return the max f1 score for all synonyms
    return max(f1_scores)

# Presence QAs will be scored using F1 across all QAs (each individual QA is just logged if it's FP FN TP TN)
def eval_presence_qa(prediction, ground_truth):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    if prediction == "yes" or prediction == "Yes":
        if ground_truth == "yes":
            true_positive = 1
        else:
            false_positive = 1
    else:
        if ground_truth == "no" or prediction == "No":
            true_negative = 1
        else:
            false_negative = 1
    return true_positive, false_positive, true_negative, false_negative

# TODO for using BLEU and ROUGE, could consider reporting geometric mean across several n-grams
def eval_dimensions_qa():
    return

# TODO
def eval_functional_performance_qa():
    return

if __name__ == '__main__':
    # # TEST RETRIEVAL QA
    # macro, all = eval_retrieval_qa('eval_metric_test_retrieval.csv')
    # print(macro)
    # print(all)
    
    # TEST COMPILATION QA
    macro_avg, all_answers = eval_compilation_qa('eval_metric_test_compilation.csv')
    print(macro_avg)
    print(all_answers)