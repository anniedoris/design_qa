import re
import pandas as pd

# Read the text from the file
file_path = 'docs/rules_pdfplumber1.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

# Initialize the DataFrame
df = pd.DataFrame(columns=['rule_number', 'rule_title', 'rule_text'])

# Iterate over the text, looking for rule numbers and extracting corresponding titles and texts
current_rule_number = None
current_rule_title = ""
current_rule_text = ""

for line in data.split('\n'):
    # if line begins with "Version 1.0 1 Sep 2023" remove it but keep the rest of the line
    if re.match(r'Version 1\.0 1 Sep 2023', line):
        line = line[22:]
    # Check if the line contains a rule number (including three-part numbers)
    match = re.match(r'([A-Z]+\.\d+(\.\d+){1,2})(.*)', line)
    if match:
        # If there is an existing rule, save it before starting a new one
        if current_rule_number:
            df = df._append({
                'rule_number': current_rule_number,
                'rule_title': current_rule_title,
                'rule_text': current_rule_text.strip()
            }, ignore_index=True)

        # Start a new rule
        current_rule_number = match.group(1).strip()
        current_rule_title = match.group(3).strip()  # The rest of the line is the title
        current_rule_text = ""
    elif current_rule_number:
        # Check if the line is part of the title (the line immediately after the rule number)
        if current_rule_title == "":
            current_rule_title = line.strip()
        else:
            # If the line is not part of the title, it's part of the rule text
            # Exclude the boilerplate text
            if not re.match(r'Formula SAE® Rules 2024 Version 1\.0 1 Sep 2023', line) and \
               not re.match(r'Formula SAE® Rules 2024 © 2023 SAE International Page \d+ of 140', line) and \
               not re.match(r'Page \d+ of 140', line) and \
               not re.match(r'^[A-Z]{1,2} - ', line) and \
               not re.match(r'^[A-Z]{1,2}\.\d ', line):
                current_rule_text += line + '\n'

# Adding the last rule if it exists
if current_rule_number:
    df = df._append({
        'rule_number': current_rule_number,
        'rule_title': current_rule_title,
        'rule_text': current_rule_text.strip()
    }, ignore_index=True)

df.loc[df['rule_number'].str.count('\.') == 3, 'rule_text'] = df['rule_title'] + " " + df['rule_text']
df.loc[df['rule_number'].str.count('\.') == 3, 'rule_title'] = ""

# save dataframe as csv
df.to_csv('docs/rules_pdfplumber1_clean1.csv', encoding='utf-8-sig', index=False)
