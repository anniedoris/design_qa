
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

# Definition QAs will be scored using F1 on bag of characters, handles synonyms
def eval_definition_qa(results_csv):
    """
    :param results_csv: the csv should contain the results from running the QA through a model.
    it should have a column called "model_prediction" and another called "ground_truth" with corresponding GT
    and a third column with "mentions" indicating the number of times the component was mentioned
    
    :returns: 
    1. overall score on QA (macro average)
    2. overall score for defintion-explicit
    3. overall score for multi-mention
    4. overall score for single mention
    5. F1 score for each QA (max score if there were synonyms)
    """
    results_df = pd.read_csv(results_csv)
    f1_scores = []
    f1_scores_definition = []
    f1_scores_multi = []
    f1_scores_single = []
    for i, row in results_df.iterrows():
        qa_f1_scores = []
        
        prediction_tokens = list(character_string_no_space(normalize_answer(row['model_prediction'])))
        synonym_tokens = []
        if ";" in row['ground_truth']:
            synonyms = row['ground_truth'].split(';')
            for syn in synonyms:
                synonym_tokens.append(list(character_string_no_space(normalize_answer(syn))))
        else:
            synonym_tokens.append(list(character_string_no_space(row['ground_truth'])))
        
        for ground_truth_tokens in synonym_tokens:
            qa_f1_scores.append(token_f1_score(prediction_tokens, ground_truth_tokens))
        
        f1_scores.append(max(qa_f1_scores))
        
        if row['mentions'] == 'definition':
            f1_scores_definition.append(max(qa_f1_scores))
        elif row['mentions'] == 'multi':
            f1_scores_multi.append(max(qa_f1_scores))
        else:
            f1_scores_single.append(max(qa_f1_scores))
    
    def mean_score(input_list):
        if len(input_list) < 1:
            return None
        else:
            return mean(input_list)
    
    return mean(f1_scores), mean_score(f1_scores_definition), mean_score(f1_scores_multi), mean_score(f1_scores_single), f1_scores

# Presence QAs will be scored using F1 across all QAs (each individual QA is just logged if it's FP FN TP TN)
def eval_presence_qa(results_csv):
    """
    :param results_csv: the csv should contain the results from running the QA through a model.
    it should have a column called "model_prediction" and another called "ground_truth" with corresponding GT
    
    :returns: 
    1. overall score on QA (macro average)
    2. F1 score for each QA
    """
    results_df = pd.read_csv(results_csv)
    
    # TODO: need to move class information over
    f1_scores = []
    f1_scores_definition = []
    f1_scores_multi = []
    f1_scores_single = []
    for i, row in results_df.iterrows():
        prediction_tokens = normalize_answer(row['model_prediction']).split()
        ground_truth_tokens = normalize_answer(row['ground_truth']).split()
        
        # Get the first yes/no in the prediction answer
        def get_first_yes_no(text_list):
            for i in text_list:
                if i == "yes":
                    return ['yes']
                if i == "no":
                    return ['no']
            return ['noanswer']
        
        prediction_tokens = get_first_yes_no(prediction_tokens)
        
        f1_scores.append(token_f1_score(prediction_tokens, ground_truth_tokens))
        
        if row['mentions'] == 'definition':
            f1_scores_definition.append(token_f1_score(prediction_tokens, ground_truth_tokens))
        elif row['mentions'] == 'multi':
            f1_scores_multi.append(token_f1_score(prediction_tokens, ground_truth_tokens))
        else:
            f1_scores_single.append(token_f1_score(prediction_tokens, ground_truth_tokens))
            
    def mean_score(input_list):
        if len(input_list) < 1:
            return None
        else:
            return mean(input_list)
    
    return mean(f1_scores), mean_score(f1_scores_definition), mean_score(f1_scores_multi), mean_score(f1_scores_single), f1_scores

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
    
    # # TEST COMPILATION QA
    # macro_avg, all_answers = eval_compilation_qa('eval_metric_test_compilation.csv')
    # print(macro_avg)
    # print(all_answers)
    
    # # TEST DEFINITION QA
    # eval_definition_qa('eval_metric_test_definition.csv')
    
    # # TEST PRESENCE QA
    # macro_avg, definitions_avg, multi_avg, single_avg, all_answers = eval_presence_qa('eval_metric_test_presence.csv')
    # print("macro average")
    # print(macro_avg)
    # print("definitions avg")
    # print(definitions_avg)
    # print("multi avg")
    # print(multi_avg)
    # print("single avg")
    # print(single_avg)
    # print("all f1")
    # print(all_answers)