from metrics import eval_functional_performance_qa


def save_results(model, macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nAll accuracies: {all_accuracies}")
    print(f"\nMacro avg bleus: {macro_avg_bleus}")
    print(f"\nAll bleus: {all_bleus}")
    print(f"\nMacro avg rogues: {macro_avg_rogues}")
    print(f"\nAll rogues: {all_rogues}")

    # Save results to txt file
    with open(f"dimension_{question_type}_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nAll accuracies: {all_accuracies}")
        text_file.write(f"\nMacro avg bleus: {macro_avg_bleus}")
        text_file.write(f"\nAll bleus: {all_bleus}")
        text_file.write(f"\nMacro avg rogues: {macro_avg_rogues}")
        text_file.write(f"\nAll rogues: {all_rogues}")


if __name__ == '__main__':
    question_type = "functional_performance"
    for model in ['gpt-4-1106-vision-preview', 'gpt-4-1106-vision-preview+RAG', 'llava-13b']:
        csv_name = f"results/dimension_{question_type}_evaluation_{model}.csv"

        # Compute the accuracy of the responses
        macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues = eval_functional_performance_qa(csv_name)

        # Print and save the results
        save_results(model, macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues)
