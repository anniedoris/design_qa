import pandas as pd
import sys

sys.path.append("../..")
from common_prompts import prompt_preamble

if __name__ == '__main__':
    qa = []
    current_image_num = 1
    question = prompt_preamble + "Also attached is an image showing six CAD views of our vehicle design." \
        " According to the rules, your analysis of the image, and any other knowledge, what is the name " \
            "of the component(s) highlighted in pink? Answer just with the name of the highlighted component(s) and nothing else."
    question_hidden = prompt_preamble + "Also attached is an image showing six CAD views of our vehicle design." \
        " According to the rules, your analysis of the image, and any other knowledge, what is the name " \
            "of the component(s) highlighted in pink? Some parts of the design have been hidden so that the " \
            "highlighted component(s) can better be visualized. Answer just with the name of the highlighted component(s) and nothing else."
    df = pd.read_csv('definitions.csv')
    for i, row in df.iterrows():
        
        if row['hidden_components'] == "yes":
            qa.append([question_hidden, row['highlighted_component'], str(current_image_num) + '.jpg'])
        else:
            qa.append([question, row['highlighted_component'], str(current_image_num) + '.jpg'])
        current_image_num += 1
        
    pd.DataFrame(qa, columns=['question', 'answer', 'image']).to_csv("../../../dataset/rule_comprehension/rule_definition_qa.csv", index=False)
    
    