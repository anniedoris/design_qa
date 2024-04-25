import csv
import sys
sys.path.append("../metrics/")
sys.path.append("../")
from metrics import eval_retrieval_qa, eval_compilation_qa

model = 'claude-opus-RAG'

question_type = 'retrieval'

csv_name = 'retrieval_evaluation_claude-opus-RAG.csv'

def save_results(model, macro_avg, all_answers, question_type):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg}")
    print(f"\nAll answers: {all_answers}")

    # Save results to txt file
    with open(f"{question_type}_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg}")
        text_file.write(f"\nAll answers: {all_answers}")

macro_avg, all_answers = eval_retrieval_qa(csv_name)

# Print and save the results
save_results(model, macro_avg, all_answers, question_type)