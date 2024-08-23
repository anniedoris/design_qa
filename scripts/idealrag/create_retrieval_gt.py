import pandas as pd

fix_gt = pd.read_csv('retrieval_evaluation_llava-13b+idealRAG5.csv')
new_gt = pd.read_csv('context_and_prompts_retrieval.csv')

for i, row in fix_gt.iterrows():
    q = row['temp']
    new_gt_row = new_gt[new_gt['prompt_without_context'] == q]['ground_truth'].values[0]
    fix_gt.loc[i, 'ground_truth'] = new_gt_row
    
fix_gt.to_csv('testing5.csv')