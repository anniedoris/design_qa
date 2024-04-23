# Compute the accuracy of the responses

import sys
sys.path.append("../metrics/")
from metrics import eval_presence_qa


def save_results(model, macro_avg, definitions_avg, multi_avg, single_avg, all_answers):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg}")
    print(f"\nDefinitions: {definitions_avg}")
    print(f"\nMulti avg: {multi_avg}")
    print(f"\nSingle avg: {single_avg}")
    print(f"\nAll answers: {all_answers}")

    # Save results to txt file
    with open(f"presence_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg}")
        text_file.write(f"\nDefinitions: {definitions_avg}")
        text_file.write(f"\nMulti avg: {multi_avg}")
        text_file.write(f"\nSingle avg: {single_avg}")
        text_file.write(f"\nAll answers: {all_answers}")

csv_name = 'presence_evaluation_claude-opus-RAG.csv'
model = 'claude-opus-RAG'

macro_avg, definitions_avg, multi_avg, single_avg, all_answers = eval_presence_qa(csv_name)

# Print and save the results
save_results(model, macro_avg, definitions_avg, multi_avg, single_avg, all_answers)