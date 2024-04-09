from metrics import eval_definition_qa


def save_results(model, macro_avg, definitions_avg, multi_avg, single_avg, all_answers):
    print(f"\nMacro avg: {macro_avg}")
    print(f"\nDefinitions: {definitions_avg}")
    print(f"\nMulti avg: {multi_avg}")
    print(f"\nSingle avg: {single_avg}")
    print(f"\nAll answers: {all_answers}")

    # Save results to txt file
    with open(f"results/definition_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg}")
        text_file.write(f"\nDefinitions: {definitions_avg}")
        text_file.write(f"\nMulti avg: {multi_avg}")
        text_file.write(f"\nSingle avg: {single_avg}")
        text_file.write(f"\nAll answers: {all_answers}")


if __name__ == '__main__':
    for model in ['gpt-4-1106-vision-preview+context', 'gpt-4-1106-vision-preview', 'llava-13b']:
        csv_name = f"results/definition_evaluation_{model}.csv"

        # Compute the accuracy of the responses
        macro_avg, definitions_avg, multi_avg, single_avg, all_answers = eval_definition_qa(csv_name)

        # Print and save the results
        save_results(model, macro_avg, definitions_avg, multi_avg, single_avg, all_answers)