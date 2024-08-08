# Compute the accuracy of the responses
import sys
sys.path.append('../metrics')
from metrics import eval_retrieval_qa, eval_compilation_qa

def save_results(model, macro_avg, all_answers, question_type):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg}")
    print(f"\nAll answers: {all_answers}")

    # Save results to txt file
    with open(f"{question_type}_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg}")
        text_file.write(f"\nAll answers: {all_answers}")

model = 'llava-13b+idealRAG'
question_type = 'compilation'

csv_name = f"{question_type}_evaluation_{model}.csv"

if question_type == 'retrieval':
    eval_presence_qa = eval_retrieval_qa
elif question_type == 'compilation':
    eval_presence_qa = eval_compilation_qa
else:
    raise ValueError("Invalid question type")
macro_avg, all_answers = eval_presence_qa(csv_name)

# Print and save the results
save_results(model, macro_avg, all_answers, question_type)
