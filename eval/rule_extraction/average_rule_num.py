import pandas as pd
from statistics import mean
from ast import literal_eval

df = pd.read_csv('compilation_evaluation_gemini-pro.csv')

total_rules = []
total_rules_gt = []
for i, row in df.iterrows():
    print("answer")
    print(row['model_prediction'])
    num_rules = len(row['model_prediction'].split(','))
    total_rules.append(num_rules)
    
    num_rules_gt = len(literal_eval(row['ground_truth']))
    total_rules_gt.append(num_rules_gt)
    print(num_rules_gt)
    # total_rules_gt.append()
    
print("AVERAGE NUM RULES")
print(mean(total_rules))

print("AVERAGE NUM RULES GT")
print(mean(total_rules_gt))