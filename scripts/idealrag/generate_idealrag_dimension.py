import pandas as pd
import re
import numpy as np
import random

RAG_LIMIT = 8716

### Only needed for extracting rule numbers ###
# Read in the definition dataframe
# dimension_df = pd.read_csv('../../dataset/rule_compliance/rule_dimension_qa/context/rule_dimension_qa_context.csv')

# rule_nums = []
# for i, row in dimension_df.iterrows():
#     rule_num = row['question'].split('does our design comply with rule')[1].split('specified')[0]
#     rule_nums.append(rule_num)
        
# dimension_df['rule_nums'] = rule_nums
# dimension_df.to_csv('dimension_context_with_rule_nums_pre.csv')

### Only needed for extracting rule numbers ###

dimension_df = pd.read_csv('dimension_context_with_rule_nums.csv')
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
    rule_nums = row['rule_nums']
    rule_nums = rule_nums.strip()
    rule_nums = rule_nums.split(',')
    rule_nums = [j.strip() for j in rule_nums]
    
    # Handle the case of just a single rule
    if len(rule_nums) == 1:
        # print(rule_nums)
        print(f"Looking for single rule num: {rule_nums}")
        rule_text = retrieve_rule_from_ground_truth(rule_nums[0], rules_gt_df)
        full_rule = rule_nums[0] + ' ' + rule_text
        extracted_context = get_random_excerpt(full_rule, RAG_LIMIT)
        ideal_rags.append(extracted_context)
        print("\n")
        
    # Handle the case of two rules (one referencing another)
    else:
        multi_rule = []
        for rule_num in rule_nums:
            print(f"Looking for multi rule num: {rule_num}")
            rule_text = retrieve_rule_from_ground_truth(rule_num, rules_gt_df)
            full_rule = rule_num + ' ' + rule_text
            multi_rule.append(get_random_excerpt(full_rule, int(np.floor(RAG_LIMIT/2))))
        joined_excerpt = "\n".join(multi_rule)
        ideal_rags.append(joined_excerpt)
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
dimension_df.to_csv('dimension_idealrag5.csv')

print(dimension_df)

# print(dimension_df)

# def mentions_in_doc(word_list, file_path):

#     # open rule document
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#     lower_case_content = content.lower()

#     master_indices = []

#     for target_phrase in word_list:

#         target_phrase = target_phrase.lower()
#         # Find all instances of a word in the document
#         # matches = re.findall(re.escape(word_list[0]), lower_case_content)
#         indices = []
#         index = lower_case_content.find(target_phrase)
#         while index != -1:
#             indices.append(index)
#             index = lower_case_content.find(target_phrase, index + 1)

#         for i in range(len(indices)):
#             # List of indices
#             start_index = indices[i]
#             end_index = indices[i] + len(target_phrase)
#             master_indices.append((start_index, end_index))

#     return len(master_indices), master_indices

# def extract_excerpt(inds, file_path, length):
#     # open rule document
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
    
#     start_ind = inds[0] - int(np.floor(length/2))
#     end_ind = start_ind + length

#     return content[start_ind:end_ind]

# def random_excerpt(file_path, length):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()

#     total_length = len(content)
#     ind_s = random.randint(0, total_length - length)
#     ind_e = ind_s + length
#     return content[ind_s:ind_e]

# # Gives the number of characters that an index list contains
# def total_char_number(inds_list, file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()

#     characters = ''
#     for inds in inds_list:
#         characters += content[inds[0]:inds[1]]
#     return len(characters)

# ideal_rag_excerpts = []

# # Do IdealRAG differently depending on the type of mention
# for i, row in definition_df.iterrows():

#     if row['mentions'] == 'mentioned':
#         target_words = row['ground_truth']

#         # Handle this specially because no direct matches
#         if target_words == "nosecone and body panels":
#             target_words = "nosecones; body panels"
#         print(f"MENTIONED Target words: {target_words}")
#         target_words = target_words.split(';')
#         target_words = [j.strip() for j in target_words]
        
#         num_mentions, inds = mentions_in_doc(target_words, '../../dataset/docs/rules_pdfplumber1.txt')

#         print(f"Num mentions: {num_mentions}")
        
#         char_count = total_char_number(inds, '../../dataset/docs/rules_pdfplumber1.txt')

#         # This is how long each segment is allowed to be
#         char_length_per_segment = int(np.floor((RAG_LIMIT - char_count)/num_mentions))

#         final_rag = ''
#         for ind_pair in inds:
#             excerpt = extract_excerpt(ind_pair, '../../dataset/docs/rules_pdfplumber1.txt', char_length_per_segment)
#             final_rag += excerpt + '\n'
        
#         ideal_rag_excerpts.append(final_rag)
    
#     if row['mentions'] == 'not_mentioned':
#         target_words = row['ground_truth']
#         print(f"NOT MENTIONED Target words: {target_words}")
#         final_rag = random_excerpt('../../dataset/docs/rules_pdfplumber1.txt', RAG_LIMIT)
#         print(f"Final len: {len(final_rag)}")
#         ideal_rag_excerpts.append(final_rag)

#     if row['mentions'] == 'definition':
#         target_words = row['ground_truth']
#         print(f"DEFINITION target words: {target_words}")

#         # open rule document
#         with open('../../dataset/docs/definitions.txt', 'r', encoding='utf-8') as file:
#             content = file.read()

#         def_length = len(content)
        
#         start_pos = random.randint(0, RAG_LIMIT - def_length)

#         first_segment_length = start_pos
#         second_segment_length = RAG_LIMIT - start_pos - def_length

#         # Add the -1 so that we can add \n
#         first_segment = random_excerpt('../../dataset/docs/rules_pdfplumber1.txt', first_segment_length - 1)
#         second_segment = random_excerpt('../../dataset/docs/rules_pdfplumber1.txt', second_segment_length - 1)

#         final_rag = first_segment + '\n' + content + '\n' + second_segment

#         print(f"Final len: {len(final_rag)}")
#         ideal_rag_excerpts.append(final_rag)

# print("######")

# # Combine the new IdealRAG with the question we are trying to ask
# full_question_plus_ideal_rag = []
# for i, rag in enumerate(ideal_rag_excerpts):
#     original_prompt = definition_df.iloc[i]['question']
#     full = original_prompt[:80] + f"Below is context from the FSAE rule document which might or might not " \
#                                 f"be relevant for the question: \n\n```\n{rag}\n```\n\n" + original_prompt[117:]
#     full_question_plus_ideal_rag.append(full)

# # Add everything together, ground truth and new questions
# definition_df['full_question'] = full_question_plus_ideal_rag

# # Swap column names
# col1 = 'question'
# col2 = 'full_question'
# definition_df = definition_df.rename(columns={col1: 'temp', col2: col1, 'temp': col2})
# definition_df.to_csv('definition_idealrag5.csv')

# print(definition_df)
        
