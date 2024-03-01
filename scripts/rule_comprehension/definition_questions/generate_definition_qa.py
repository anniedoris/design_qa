import pandas as pd
import sys

sys.path.append("../..")
from common_prompts import prompt_preamble

if __name__ == '__main__':
    qa = []
    current_image_num = 1
    common_question1 = "Also attached is an image showing six CAD views of our vehicle design." \
        " What is the name of the component(s) highlighted in pink?"
    common_question2 = " Answer just with the name of the highlighted component(s) and nothing else."
    question = prompt_preamble + common_question1 + common_question2
    question_hidden = prompt_preamble + common_question1 + " Some parts of the design have been hidden so that the " \
            "highlighted component(s) can better be visualized." + common_question2
    df = pd.read_csv('definitions_raw.csv')
    for i, row in df.iterrows():
        
        if row['hidden_components'] == "yes":
            qa.append([question_hidden, row['highlighted_component'], str(current_image_num) + '.jpg', row['mention_in_rules']])
        else:
            qa.append([question, row['highlighted_component'], str(current_image_num) + '.jpg', row['mention_in_rules']])
        current_image_num += 1
        
    pd.DataFrame(qa, columns=['question', 'ground_truth', 'image', 'mentions']).to_csv("../../../dataset/rule_comprehension/rule_definition_qa.csv", index=False)
    
    