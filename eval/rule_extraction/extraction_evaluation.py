from metrics import eval_retrieval_qa, eval_compilation_qa


def save_results(model, macro_avg, all_answers, question_type):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg}")
    print(f"\nAll answers: {all_answers}")

    # Save results to txt file
    with open(f"{question_type}/results/{question_type}_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg}")
        text_file.write(f"\nAll answers: {all_answers}")


if __name__ == '__main__':
    for question_type in ['retrieval', "compilation"]:
        # models available: 'gpt-4-0125-preview+RAG', 'gpt-4-0125-preview', 'llama-2-70b-chat', 'llava-13b', 'gpt-4-1106-vision-preview+RAG', 'gpt-4-1106-vision-preview'
        for model in ['llava-13b', 'gpt-4-1106-vision-preview+RAG', 'gpt-4-1106-vision-preview']:
            csv_name = f"{question_type}/results/{question_type}_evaluation_{model}.csv"

            # Compute the accuracy of the responses
            if question_type == 'retrieval':
                eval_presence_qa = eval_retrieval_qa
            elif question_type == 'compilation':
                eval_presence_qa = eval_compilation_qa
            else:
                raise ValueError("Invalid question type")
            macro_avg, all_answers = eval_presence_qa(csv_name)

            # Print and save the results
            save_results(model, macro_avg, all_answers, question_type)
