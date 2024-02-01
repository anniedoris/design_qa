import pandas as pd 

def remove_problematic_characters(rule_text):
    problem = "â"
    apostrophe = "¢"
    
    print(rule_text)

    if problem in rule_text:
        print("Here")

        apostrophe_indices 
        for i, char in enumerate(rule_text):
            print(i)
            if i != len(rule_text) - 1:
                if rule_text[i + 1] == apostrophe:
        
        
        
        new_string = rule_text[:i] + "-" + rule_text[i+2:]

    print(new_string)
    return rule_text

df = pd.read_csv('rules_pdfplumber1_clean1.csv', encoding='unicode_escape')

remove_problematic_characters(df.iloc[100]['rule_text'])