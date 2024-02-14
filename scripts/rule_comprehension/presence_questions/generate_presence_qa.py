import pandas as pd
import sys

sys.path.append("../..")
from common_prompts import prompt_preamble

if __name__ == '__main__':
    qa = []
    current_image_num = 1
    # common_question1 = "Also attached is an image showing six CAD views of our vehicle design." \
    #     " What is the name of the component(s) highlighted in pink?"
    # common_question2 = " Answer just with the name of the highlighted component(s) and nothing else."
    # question = prompt_preamble + common_question1 + common_question2
    # question_hidden = prompt_preamble + common_question1 + " Some parts of the design have been hidden so that the " \
    #         "highlighted component(s) can better be visualized." + common_question2
            
            
    # question = prompt_preamble + "Also attached is an image showing seven CAD views (each boxed in black) of our vehicle design. The top, big view " \
    #     "shows a close-up view of the design. The six smaller views on the bottom of the image show different complete views of the CAD  " \
    #     "of the vehicle and are provided for context. Note that the close-up view orientation matches one of the six complete view orientations. Looking at the close-up view, is the " \
    #     + XX + " visible in the close-up view? Answer simply with yes or no."
        
    df = pd.read_csv('presence_raw.csv')
    for i, row in df.iterrows():
        
        all_ground_truth_components = row['component'].split(';')
        
        if len(all_ground_truth_components) > 1:
            present_components = "the " + (" or the").join(all_ground_truth_components)
        else:
            present_components = "the " + all_ground_truth_components[0]
        
        question = prompt_preamble + "Also attached is an image showing seven CAD views (each boxed in black) of our vehicle design. The top, big view " \
        "shows a close-up view of the design. The six smaller views on the bottom of the image show different complete views of the CAD  " \
        "of the vehicle and are provided for context. Note that the close-up view orientation matches one of the six complete view orientations. " \
        "The close-up view may also have some components hidden (with respect to the corresponding complete view) for visualization of specific components. " \
        "Looking at the close-up view, is/are " + present_components + " visible in the close-up view? Answer simply with yes or no."
        
        # TODO: add mentions in rules back like definition_qa
        qa.append([question, row['present'], str(current_image_num) + '.jpg'])
        
        current_image_num += 1
        
    pd.DataFrame(qa, columns=['question', 'answer', 'image']).to_csv("../../../dataset/rule_comprehension/rule_presence_qa.csv", index=False)
    
    