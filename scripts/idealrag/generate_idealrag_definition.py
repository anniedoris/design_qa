import pandas as pd
import re
import numpy as np
import random

RAG_LIMIT = 8716

# Read in the definition dataframe
definition_df = pd.read_csv('../../dataset/rule_comprehension/rule_definition_qa.csv')

print(definition_df)

def mentions_in_doc(word_list, file_path):

    # open rule document
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    lower_case_content = content.lower()

    master_indices = []

    for target_phrase in word_list:

        target_phrase = target_phrase.lower()
        # Find all instances of a word in the document
        # matches = re.findall(re.escape(word_list[0]), lower_case_content)
        indices = []
        index = lower_case_content.find(target_phrase)
        while index != -1:
            indices.append(index)
            index = lower_case_content.find(target_phrase, index + 1)

        for i in range(len(indices)):
            # List of indices
            start_index = indices[i]
            end_index = indices[i] + len(target_phrase)
            master_indices.append((start_index, end_index))

    return len(master_indices), master_indices

def extract_excerpt(inds, file_path, length):
    # open rule document
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    start_ind = inds[0] - int(np.floor(length/2))
    end_ind = start_ind + length

    return content[start_ind:end_ind]

def random_excerpt(file_path, length):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    total_length = len(content)
    ind_s = random.randint(0, total_length - length)
    ind_e = ind_s + length
    return content[ind_s:ind_e]

# Gives the number of characters that an index list contains
def total_char_number(inds_list, file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    characters = ''
    for inds in inds_list:
        characters += content[inds[0]:inds[1]]
    return len(characters)

ideal_rag_excerpts = []

# Do IdealRAG differently depending on the type of mention
for i, row in definition_df.iterrows():

    if row['mentions'] == 'mentioned':
        target_words = row['ground_truth']

        # Handle this specially because no direct matches
        if target_words == "nosecone and body panels":
            target_words = "nosecones; body panels"
        print(f"MENTIONED Target words: {target_words}")
        target_words = target_words.split(';')
        target_words = [j.strip() for j in target_words]
        
        num_mentions, inds = mentions_in_doc(target_words, '../../dataset/docs/rules_pdfplumber1.txt')

        print(f"Num mentions: {num_mentions}")
        
        char_count = total_char_number(inds, '../../dataset/docs/rules_pdfplumber1.txt')

        # This is how long each segment is allowed to be
        char_length_per_segment = int(np.floor((RAG_LIMIT - char_count)/num_mentions))

        final_rag = ''
        for ind_pair in inds:
            excerpt = extract_excerpt(ind_pair, '../../dataset/docs/rules_pdfplumber1.txt', char_length_per_segment)
            final_rag += excerpt + '\n'
        
        ideal_rag_excerpts.append(final_rag)
    
    if row['mentions'] == 'not_mentioned':
        target_words = row['ground_truth']
        print(f"NOT MENTIONED Target words: {target_words}")
        final_rag = random_excerpt('../../dataset/docs/rules_pdfplumber1.txt', RAG_LIMIT)
        print(f"Final len: {len(final_rag)}")
        ideal_rag_excerpts.append(final_rag)

    if row['mentions'] == 'definition':
        target_words = row['ground_truth']
        print(f"DEFINITION target words: {target_words}")

        # open rule document
        with open('../../dataset/docs/definitions.txt', 'r', encoding='utf-8') as file:
            content = file.read()

        def_length = len(content)
        
        start_pos = random.randint(0, RAG_LIMIT - def_length)

        first_segment_length = start_pos
        second_segment_length = RAG_LIMIT - start_pos - def_length

        # Add the -1 so that we can add \n
        first_segment = random_excerpt('../../dataset/docs/rules_pdfplumber1.txt', first_segment_length - 1)
        second_segment = random_excerpt('../../dataset/docs/rules_pdfplumber1.txt', second_segment_length - 1)

        final_rag = first_segment + '\n' + content + '\n' + second_segment

        print(f"Final len: {len(final_rag)}")
        ideal_rag_excerpts.append(final_rag)

print("######")

# Combine the new IdealRAG with the question we are trying to ask
full_question_plus_ideal_rag = []
for i, rag in enumerate(ideal_rag_excerpts):
    original_prompt = definition_df.iloc[i]['question']
    full = original_prompt[:80] + f"Below is context from the FSAE rule document which might or might not " \
                                f"be relevant for the question: \n\n```\n{rag}\n```\n\n" + original_prompt[117:]
    full_question_plus_ideal_rag.append(full)

# Add everything together, ground truth and new questions
definition_df['full_question'] = full_question_plus_ideal_rag

# Swap column names
col1 = 'question'
col2 = 'full_question'
definition_df = definition_df.rename(columns={col1: 'temp', col2: col1, 'temp': col2})
definition_df.to_csv('definition_idealrag5.csv')

print(definition_df)
        
