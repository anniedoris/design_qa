import pandas as pd

# This script gets a list of rules for the specified rule category (e.g. "D" is Dynamic Event rules).
# Output is .txt in docs folder

rule_category = "D"
file_path = '../../dataset/docs/rules_pdfplumber1.txt' # File to read rules from
rule_numbers = []

with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

for i, line in enumerate(data.split('\n')):

    # None of the rule names should have fewer than 2 characters (and there were som)
    if len(line) > 2:
        if line !='' and line[0:len(rule_category)] == rule_category and line[len(rule_category)] == '.':
            print("found rule")
            rule_num = line.split(' ')[0]
            rule_numbers.append(rule_num)

with open('../../dataset/docs/rule_nums/' + rule_category + ".txt", "w") as file:
    for rule in rule_numbers:
        file.write(rule + '\n')