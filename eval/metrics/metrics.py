import re
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import pandas as pd
from statistics import mean
import ast


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

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1.,))  # BELU-1
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
        try:
            prediction_tokens = row['model_prediction'].split(", ")
        except Exception as e:
            print(f"Error: {e}")
            print("Model output non-standard")
        ground_truth_tokens = ast.literal_eval(row['ground_truth'])
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
    3. overall score for mentioned
    4. overall score for not mentioned
    5. F1 score for each QA (max score if there were synonyms)
    """
    results_df = pd.read_csv(results_csv)
    f1_scores = []
    f1_scores_definition = []
    f1_scores_mentioned = []
    f1_scores_none = []
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
        elif row['mentions'] == 'mentioned':
            f1_scores_mentioned.append(max(qa_f1_scores))
        else:
            f1_scores_none.append(max(qa_f1_scores))

    def mean_score(input_list):
        if len(input_list) < 1:
            return None
        else:
            return mean(input_list)

    return mean(f1_scores), mean_score(f1_scores_definition), mean_score(f1_scores_mentioned), mean_score(
        f1_scores_none), f1_scores


# Presence QAs will be scored using accuracy
def eval_presence_qa(results_csv):
    """
    :param results_csv: the csv should contain the results from running the QA through a model.
    it should have a column called "model_prediction" and another called "ground_truth" with corresponding GT
    
    :returns: 
    1. overall score on QA (macro average)
     2. overall score for defintion-explicit
    3. overall score for mentioned
    4. overall score for not mentioned
    5. F1 score for each QA (max score if there were synonyms)
    """
    results_df = pd.read_csv(results_csv)

    # Lists ready for scores
    f1_scores = []
    f1_scores_definition = []
    f1_scores_mentioned = []
    f1_scores_not_mentioned = []
    for i, row in results_df.iterrows():
        prediction_tokens = normalize_answer(row['model_prediction']).split()
        ground_truth_tokens = normalize_answer(row['ground_truth']).split()

        # Extract the first yes/no in the prediction answer, this will be what we will score the model on
        def get_first_yes_no(text_list):
            for i in text_list:
                if i == "yes":
                    return ['yes']
                if i == "no":
                    return ['no']
            return ['noanswer']

        prediction_tokens = get_first_yes_no(prediction_tokens)

        # Computing f1_score between yeses and nos on a word level will produce 1 if the responses agree and 0
        # if the responses don't. So this will produce list of 1s and 0s across all questions
        f1_scores.append(token_f1_score(prediction_tokens, ground_truth_tokens))

        # Log results dependening on number of mentions
        if row['mentions'] == 'definition':
            f1_scores_definition.append(token_f1_score(prediction_tokens, ground_truth_tokens))
        elif row['mentions'] == 'mentioned':
            f1_scores_mentioned.append(token_f1_score(prediction_tokens, ground_truth_tokens))
        else:
            f1_scores_not_mentioned.append(token_f1_score(prediction_tokens, ground_truth_tokens))

    def mean_score(input_list):
        if len(input_list) < 1:
            return None
        else:
            return mean(input_list)

    # Return means
    return mean(f1_scores), mean_score(f1_scores_definition), mean_score(f1_scores_mentioned), mean_score(
        f1_scores_not_mentioned), f1_scores


def eval_dimensions_qa(results_csv):
    """
    :param results_csv: the csv should contain the results from running the QA through a model.
    it should have a column called "model_prediction" and another called "ground_truth" with corresponding GT
    
    :returns: 
    1. overall accuracy score on QAs (macro average)
    2. macro avg accuracy for direct dimension qas
    3. macro avg accuracy for scale bar qas
    4. all accuracies
    5. macro avg bleu-2
    6. all bleu-2 scores
    7. macro avg rogue-l
    8. all rogue-l scores
    """
    results_df = pd.read_csv(results_csv)

    # Compute accuracies
    accuracies = []
    direct_dim_accuracies = []
    scale_bar_accuracies = []
    bleus = []
    rogues = []
    
    for i, row in results_df.iterrows():
        ground_truth_tokens = normalize_answer(row['ground_truth']).split()
            
        # Locate the explanation portion of the answer
        def find_explanation_and_answer(pred):
            explanation = ""
            answer = ""
            explanation_index = None
            answer_index = None
            text_list = pred.lower().split()
            for i, word in enumerate(text_list):
                if "explanation:" in word:
                    explanation_index = i
                if "answer:" in word:
                    answer_index = i
            if explanation_index != None:
                explanation = " ".join(pred.split()[explanation_index + 1 : answer_index])
            if answer_index != None:
                answer = " ".join(pred.split()[answer_index + 1:])
                if len(answer.split()) > 3:
                    answer = " ".join(answer.split()[:3])
                    
            return answer, explanation
        
        answer, explanation = find_explanation_and_answer(row['model_prediction'])
        
        # Extract the first yes/no in the prediction answer, this will be what we will score the model on
        def get_first_yes_no(text_list):
            for i in text_list:
                if i == "yes":
                    return ['yes']
                if i == "no":
                    return ['no']
            return ['noanswer']
        
        prediction_yes_no = get_first_yes_no(normalize_answer(answer).split())
        
        # Computing f1_score between yeses and nos on a word level will produce 1 if the responses agree and 0
        # if the responses don't. So this will produce list of 1s and 0s across all questions
        accuracies.append(token_f1_score(prediction_yes_no, ground_truth_tokens))
        
        if row['dimension_type'] == "direct":
            direct_dim_accuracies.append(token_f1_score(prediction_yes_no, ground_truth_tokens))
        else:
            scale_bar_accuracies.append(token_f1_score(prediction_yes_no, ground_truth_tokens))
        
        # Only have explanations for direct dimensions
        if row['dimension_type'] == "direct":
            
            if explanation == "":
                rogues.append(0)
                bleus.append(0)
            else:
                # compute bleu-2
                bleu_2 = bleu_score(row['explanation'], explanation, 2) #used bleu 2 instead of 4 because only single reference leads to lower
                # chance of n-gram overlap
                bleus.append(bleu_2)
                
                # compute rouge-l
                rouge_l = score_rouge(row['explanation'], explanation)
                rogues.append(rouge_l)

    def mean_score(input_list):
        if len(input_list) < 1:
            return None
        else:
            return mean(input_list)

    return mean_score(accuracies), mean_score(direct_dim_accuracies), mean_score(scale_bar_accuracies), accuracies, mean_score(bleus), bleus, mean(rogues), rogues


def eval_functional_performance_qa(results_csv):
    """
        :param results_csv: the csv should contain the results from running the QA through a model.
        it should have a column called "model_prediction" and another called "ground_truth" with corresponding GT
        
        :returns: 
        1. overall accuracy score on QAs (macro average)
        2. all accuracies
        3. macro avg bleu-2
        4. all bleu-2 scores
        5. macro avg rogue-l
        6. all rogue-l scores
    """
    results_df = pd.read_csv(results_csv)

    # Compute accuracies
    accuracies = []
    bleus = []
    rogues = []
    
    for i, row in results_df.iterrows():
        ground_truth_tokens = normalize_answer(row['ground_truth']).split()
            
        # Locate the explanation portion of the answer
        def find_explanation_and_answer(pred):
            explanation = ""
            answer = ""
            explanation_index = None
            answer_index = None
            text_list = pred.lower().split()
            for i, word in enumerate(text_list):
                if "explanation:" in word:
                    explanation_index = i
                if "answer:" in word:
                    answer_index = i
            if explanation_index != None:
                explanation = " ".join(pred.split()[explanation_index + 1 : answer_index])
            if answer_index != None:
                answer = " ".join(pred.split()[answer_index + 1:])
                if len(answer.split()) > 3:
                    answer = " ".join(answer.split()[:3])
            return answer, explanation
        
        answer, explanation = find_explanation_and_answer(row['model_prediction'])
        
        # Extract the first yes/no in the prediction answer, this will be what we will score the model on
        def get_first_yes_no(text_list):
            for i in text_list:
                if i == "yes":
                    return ['yes']
                if i == "no":
                    return ['no']
            return ['noanswer']
        
        prediction_yes_no = get_first_yes_no(normalize_answer(answer).split())
        
        # Computing f1_score between yeses and nos on a word level will produce 1 if the responses agree and 0
        # if the responses don't. So this will produce list of 1s and 0s across all questions
        accuracies.append(token_f1_score(prediction_yes_no, ground_truth_tokens))
            
        if explanation == "":
            rogues.append(0)
            bleus.append(0)
        else:
            # compute bleu-2
            bleu_2 = bleu_score(row['explanation'], explanation, 2) #used bleu 2 instead of 4 because only single reference leads to lower
            # chance of n-gram overlap
            bleus.append(bleu_2)
            
            # compute rouge-l
            rouge_l = score_rouge(row['explanation'], explanation)
            rogues.append(rouge_l)

    def mean_score(input_list):
        if len(input_list) < 1:
            return None
        else:
            return mean(input_list)

    return mean_score(accuracies), accuracies, mean_score(bleus), bleus, mean(rogues), rogues


if __name__ == '__main__':
    print("Testing evaluation metrics")
    # # TEST RETRIEVAL QA
    # macro, all = eval_retrieval_qa('eval_metric_test_retrieval.csv')
    # print(macro)
    # print(all)

    # # TEST COMPILATION QA
    # print(pd.read_csv('../rule_extraction/compilation_evaluation_gpt4.csv')['ground_truth'].head())
    # print(pd.read_csv('../rule_extraction/compilation_evaluation_gpt4.csv')['model_prediction'].head())
    # macro_avg, all_answers = eval_compilation_qa('../rule_extraction/compilation_evaluation_gpt4.csv')
    # print(macro_avg)
    # print(all_answers)
    # macro_avg, all_answers = eval_compilation_qa('eval_metric_test_compilation.csv')
    # print(macro_avg)
    # print(all_answers)

    # # TEST DEFINITION QA
    # macro_avg, definitions_avg, mentioned_avg, no_mention_avg, all_answers = eval_definition_qa('eval_metric_test_definition.csv')
    # print("Macro avg")
    # print(macro_avg)
    # print("Definitions")
    # print(definitions_avg)
    # print("Mentioned avg")
    # print(mentioned_avg)
    # print("No mention avg")
    # print(no_mention_avg)
    # print("All answers")
    # print(all_answers)

    # # TEST PRESENCE QA
    # macro_avg, definitions_avg, mentioned_avg, no_mentioned_avg, all_answers = eval_presence_qa('eval_metric_test_presence.csv')
    # print("macro average")
    # print(macro_avg)
    # print("definitions avg")
    # print(definitions_avg)
    # print("mentioned avg")
    # print(mentioned_avg)
    # print("no mentioned avg")
    # print(no_mentioned_avg)
    # print("all f1")
    # print(all_answers)
    
    # # # TEST DIMENSION QA
    # macro_avg_accuracy, direct_dim_avg, scale_bar_avg, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues = eval_dimensions_qa('compliance_test.csv')
    # eval_dimensions_qa('compliance_test.csv')
    # print("macro average")
    # print(macro_avg_accuracy)
    # print("direct dimension accuracies")
    # print(direct_dim_avg)
    # print("scale bar average")
    # print(scale_bar_avg)
    # print("all accuracies")
    # print(all_accuracies)
    # print("macro avg bleus")
    # print(macro_avg_bleus)
    # print("all_bleus")
    # print(all_bleus)
    # print("macro avg rogues")
    # print(macro_avg_rogues)
    # print("all_rogues")
    # print(all_rogues)
    
    # # TEST FP QA
    macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues = eval_functional_performance_qa('compliance_test_fp.csv')
    eval_dimensions_qa('compliance_test.csv')
    print("macro average")
    print(macro_avg_accuracy)
    print("all accuracies")
    print(all_accuracies)
    print("macro avg bleus")
    print(macro_avg_bleus)
    print("all_bleus")
    print(all_bleus)
    print("macro avg rogues")
    print(macro_avg_rogues)
    print("all_rogues")
    print(all_rogues)
    
