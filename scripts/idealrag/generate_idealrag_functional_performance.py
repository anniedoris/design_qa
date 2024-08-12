import pandas as pd
import re
import numpy as np
import random

RAG_LIMIT = 8716


### Only needed for extracting rule numbers ###
## Read in the definition dataframe
# if q_type == 'context':
#     dimension_df = pd.read_csv('../../dataset/rule_compliance/rule_dimension_qa/context/rule_dimension_qa_context.csv')
# else:
#     dimension_df = pd.read_csv('../../dataset/rule_compliance/rule_dimension_qa/detailed_context/rule_dimension_qa_detailed_context.csv')

# rule_nums = []
# for i, row in dimension_df.iterrows():
#     rule_num = row['question'].split('does our design comply with rule')[1].split('specified')[0]
#     rule_nums.append(rule_num)
        
# dimension_df['rule_nums'] = rule_nums
# dimension_df.to_csv('dimension_with_rule_nums_pre.csv')

### Only needed for extracting rule numbers ###

dimension_df = pd.read_csv('functional_performance_with_rule_nums.csv')

rules_gt_df = pd.read_csv('context_and_prompts_compilation.csv')

# Extract out the rule number
rules_gt_df['rule_num'] = rules_gt_df['prompt_without_context'].apply(lambda x: x.split('What does rule')[1].split('state exactly')[0].strip())

def retrieve_rule_from_ground_truth(rule_num, gt_df):
    gt_rule = gt_df[gt_df['rule_num'] == rule_num]['ground_truth'].values[0]
    return gt_rule

def get_random_excerpt(rule_to_contain, total_excerpt_length):
     # open rule document
    with open('../../dataset/docs/rules_pdfplumber1.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    rule_index = content.find(rule_to_contain)
    if rule_index == -1:
        print("GT RULE NOT FOUND IN RULE DOC")

    # Randomly place the rule in the excerpt
    loc_rule = random.randint(0, total_excerpt_length - len(rule_to_contain))
    start_ind = rule_index - loc_rule
    end_ind = start_ind + total_excerpt_length
    found_excerpt = content[start_ind:end_ind]
    return found_excerpt

ideal_rags = []
for i, row in dimension_df.iterrows():
    rule_nums = row['rule_num']
    rule_nums = rule_nums.strip()
    print(rule_nums)

    # Handle the case of just a single rule

    # print(rule_nums)
    print(f"Looking for single rule num: {rule_nums}")
    rule_text = retrieve_rule_from_ground_truth(rule_nums, rules_gt_df)
    full_rule = rule_nums + ' ' + rule_text
    extracted_context = get_random_excerpt(full_rule, RAG_LIMIT)
    ideal_rags.append(extracted_context)
    print("\n")

# Combine the new IdealRAG with the question we are trying to ask
full_question_plus_ideal_rag = []
for i, rag in enumerate(ideal_rags):
    original_prompt = dimension_df.iloc[i]['question']
    full = original_prompt[:80] + f"Below is context from the FSAE rule document which might or might not " \
                                f"be relevant for the question: \n\n```\n{rag}\n```\n\n" + original_prompt[117:]
    full_question_plus_ideal_rag.append(full)

# Add everything together, ground truth and new questions
dimension_df['full_question'] = full_question_plus_ideal_rag

# Swap column names
col1 = 'question'
col2 = 'full_question'
dimension_df = dimension_df.rename(columns={col1: 'temp', col2: col1, 'temp': col2})

dimension_df.to_csv('functional_performance_idealrag5.csv')

print(dimension_df)
