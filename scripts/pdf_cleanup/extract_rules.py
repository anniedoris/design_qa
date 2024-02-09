
import pandas as pd

# Get rule category you want to extract
rule_section = 'D'
rule_texts = []

# Get the rule numbers from the rule num text file
all_rules = []
with open('../../dataset/docs/rule_nums/' + rule_section + '_rule_nums.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        all_rules.append(line)
all_rules = [i.strip('\n') for i in lines]

# Read the appropriate rule section document
file_path = '../../dataset/docs/rule_section_text/' + rule_section + '_rules' + '.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

rule_text_by_line = data.split('\n')

# Go through each rule number in the rule number list
for rule_ind, rule_num in enumerate(all_rules[:-1]):

    print(f"RULE NUMBER: {rule_num}")

    # For populating the rule text
    rule_text = ''

    next_rule_found = False
    rule_found = False
    rule_start_index = None
    next_rule_num = all_rules[rule_ind+1]

    # Look for rule start point in the doc section
    while not rule_found:
        for i, line in enumerate(rule_text_by_line):
            if line[0:len(rule_num) + 1] == rule_num + ' ':
                rule_found = True # indicate that rule has been found
                rule_start_index = i + 1
                rule_text += line.split(rule_num + ' ')[1] + '\n'
                break

    # Search for the next rule
    while not next_rule_found:
        for line in rule_text_by_line[rule_start_index:]:
            if line[0:len(next_rule_num) + 1] == next_rule_num + ' ':
                next_rule_found = True # indicate next rule is found so end of rule text
                break

            # Otherwise we haven't reached the end of rule and keep adding text
            else:
                if 'Formula SAE® Rules 2024 © 2023' not in line and 'Version 1.0 1 Sep 2023' not in line:
                    rule_text += line + '\n'

    rule_texts.append(rule_text.strip('\n'))

df = pd.DataFrame({'rule_num': all_rules[:-1], 'rule_text': rule_texts})
df.to_csv('../../dataset/docs/csv_rules/' + rule_section + '_extracted.csv')
