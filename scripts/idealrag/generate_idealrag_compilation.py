import pandas as pd
import ast
import numpy as np
import random

RAG_LIMIT = 8716

# Read in the compilation question lists
compilation_df = pd.read_csv('../../dataset/rule_extraction/rule_compilation_qa.csv')

# Read in the ground truth rule for each list
gt = pd.read_csv('context_and_prompts_compilation.csv')

# Extract out the rule number
gt['rule_num'] = gt['prompt_without_context'].apply(lambda x: x.split('What does rule')[1].split('state exactly')[0].strip())

# Find the length (character count) of all the rules in a specific compilation list
def total_text_length(rule_text):
    text_total = ''
    for t in rule_text:
        text_total = text_total + t
    return len(text_total)

# Remove the longest rule from the list of rules
def remove_longest_string(strings, rule_list):
    if not strings:
        return strings, rule_list
    # Find the index of the longest string
    longest_index = max(range(len(strings)), key=lambda i: len(strings[i]))
    # Remove the elements at that index from both lists
    strings.pop(longest_index)
    rule_list.pop(longest_index)
    return strings, rule_list

# Will contain the good RAG segments
good_context = []

# Go through each rule in the compilation list, and get the rule text
for i, row in compilation_df.iterrows():
    print(f"COMPILATION: {i}")
    rule_list = ast.literal_eval(row['ground_truth'])
    rule_text = []
    for rule in rule_list:
        specific_text = gt[gt['rule_num'] == rule]

        # Check for cases where there is no match for the ground truth rule. NOTE: I manually added all missing "title" rules in context_and_prompts_compilation.csv
        # so there should be no print outs for missing rules
        if specific_text.empty:
            specific_text = ''
            print(f"Missing rule: {rule}")
        else:
            specific_text = specific_text.ground_truth.values[0]
        rule_text.append(specific_text)
    
    # This appends the rule number to the rule text
    rag_text = []
    for i, gt_rule in enumerate(rule_text):
        rag_text.append(rule_list[i] + ' ' + gt_rule)

    # Remove rules from the list until it abides by the appropriate character count, 10895 is the average number of characters for the retrieval RAG context
    # Take the limit to be 10895 minus the number of rules in the list (so we can add \n between the different rules)
    limit = RAG_LIMIT - len(rule_list)

    # If the rule text is too long, remove the longest rules (so we preserve the largest number of rules)
    mod_count = 0
    while total_text_length(rag_text) > limit:
        rag_text, rule_list = remove_longest_string(rag_text, rule_list)
        mod_count +=1


    final_length = total_text_length(rag_text)
    num_rules = len(rule_list)

    # If we haven't hit the character limit, add in more random rules so we hit the character limit
    if mod_count == 0:
        while total_text_length(rag_text) < RAG_LIMIT:
            random_row = gt.sample(n=1)
            random_rule_text = random_row['ground_truth'].values[0]
            random_rule_num = random_row['rule_num'].values[0]
            random_rule = random_rule_num + ' ' + random_rule_text
            rag_text.append(random_rule)
        
        while total_text_length(rag_text) > limit:
            rag_text.pop()
    
    # Randomly shuffle the rules
    random.shuffle(rag_text)

    # Join the shuffled rules together into one text blurb
    ideal_rag = '\n'.join(rag_text)

    # Check limit. 
    # TODO: there were cases (i'm not sure why) where we were surpassing the limit. For these, I ran this script several times and replaced
    # "bad", over the limit ideal rag sections with under the limit ideal rag sections (from a different run)
    if len(ideal_rag) > RAG_LIMIT:
        print("bad")

    good_context.append(ideal_rag)

# Combine the new IdealRAG with the question we are trying to ask
full_question_plus_ideal_rag = []
for i, rag in enumerate(good_context):
    original_prompt = compilation_df.iloc[i]['question']
    full = original_prompt[:80] + f"Below is context from the FSAE rule document which might or might not " \
                                f"be relevant for the question: \n\n```\n{rag}\n```\n\n" + original_prompt[117:]
    full_question_plus_ideal_rag.append(full)

# Add everything together, ground truth and new questions
compilation_df['full_question'] = full_question_plus_ideal_rag

# Swap column names
col1 = 'question'
col2 = 'full_question'
compilation_df = compilation_df.rename(columns={col1: 'temp', col2: col1, 'temp': col2})
compilation_df.to_csv('compilation_idealrag1.csv')

print(compilation_df)

