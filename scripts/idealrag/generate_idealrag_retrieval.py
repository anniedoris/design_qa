import pandas as pd
import random

RAG_LIMIT = 8716

def get_random_excerpt(rule_to_contain, total_excerpt_length):
     # open rule document
    with open('../../dataset/docs/rules_pdfplumber1.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    rule_index = content.find(rule_to_contain)
    if rule_index == -1:
        print("GT RULE NOT FOUND IN RULE DOC")
        print(rule_to_contain)

    # Randomly place the rule in the excerpt
    found_excerpt = ''
    while len(found_excerpt) < RAG_LIMIT: # This handles the rules at the end of the document
        loc_rule = random.randint(0, total_excerpt_length - len(rule_to_contain))
        start_ind = rule_index - loc_rule
        end_ind = start_ind + total_excerpt_length
        found_excerpt = content[start_ind:end_ind]
    return found_excerpt

# Read in the retrieval question lists
# Took out EV.5.51 and EV.7.2.1 bc issues finding them verbatim
# TODO: update ground truth rules in this doc to be those in context and prompts retrieval for scoring
retrieval_df = pd.read_csv('rule_retrieval_qa_original_minus_two.csv')

# Read in the ground truth rule for each list
rules_gt_df = pd.read_csv('context_and_prompts_retrieval.csv')

# Extract the rule numbers 
rules_gt_df['rule_num'] = rules_gt_df['prompt_without_context'].apply(lambda x: x.split('What does rule')[1].split('state exactly')[0].strip())

all_context = []
for i, row in retrieval_df.iterrows():
    rule_num = row['question']
    rule_num = rule_num.split('What does rule ')[1].split('state exactly')[0].strip()
    rule_text = rules_gt_df[rules_gt_df['rule_num'] == rule_num]['ground_truth'].values[0]
    full_rule = rule_num + ' ' + rule_text
    
    guaranteed_rag_excerpt = get_random_excerpt(full_rule, RAG_LIMIT)
    all_context.append(guaranteed_rag_excerpt)
    
# Combine the new IdealRAG with the question we are trying to ask
full_question_plus_ideal_rag = []
for i, rag in enumerate(all_context):
    original_prompt = retrieval_df.iloc[i]['question']
    full = original_prompt[:80] + f"Below is context from the FSAE rule document which might or might not " \
                                f"be relevant for the question: \n\n```\n{rag}\n```\n\n" + original_prompt[117:]
    full_question_plus_ideal_rag.append(full)

# Add everything together, ground truth and new questions
retrieval_df['full_question'] = full_question_plus_ideal_rag

# Swap column names
col1 = 'question'
col2 = 'full_question'
retrieval_df = retrieval_df.rename(columns={col1: 'temp', col2: col1, 'temp': col2})
retrieval_df.to_csv('retrieval_idealrag1.csv')

print(retrieval_df)