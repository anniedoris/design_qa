
import re
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

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
## Text F1, bag of tokens
########################
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

def bleu_score(reference, hypothesis, gram):
    """
    Taken from the ScienceQA evaluations.py script
    """

    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

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
# Retrieval QAs will be scored using F1
def eval_retrieval_qa(prediction, ground_truth):
    return token_f1_score(prediction, ground_truth)

# TODO
def eval_compilation_qa():
    return

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

# TODO
def eval_dimensions_qa():
    return

# TODO
def eval_functional_performance_qa():
    return

if __name__ == '__main__':
    test_a = "This is a match"
    test_b = "This is not a match"
    score = bleu_score(test_a, test_b, 1)
    # score = eval_presence_qa(test_a, test_b)
    print(score)