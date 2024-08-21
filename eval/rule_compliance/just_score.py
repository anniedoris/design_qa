
import sys
sys.path.append('../metrics')
from metrics import eval_functional_performance_qa
from metrics import eval_dimensions_qa

csv_name = 'dimension_context_evaluation_llava-13b.csv'
# Options are functional_performance, detailed_context, or context
question_type = 'context'
model = 'llava-13b'

def save_results_functional_performance(model, macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues, macro_similarity, all_similarities):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nAll accuracies: {all_accuracies}")
    print(f"\nMacro avg bleus: {macro_avg_bleus}")
    print(f"\nAll bleus: {all_bleus}")
    print(f"\nMacro avg rogues: {macro_avg_rogues}")
    print(f"\nAll rogues: {all_rogues}")
    print(f"\nMacro avg similarities: {macro_similarity}")
    print(f"\nAll rogues: {all_similarities}")

    # Save results to txt file
    with open(f"dimension_{question_type}_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nAll accuracies: {all_accuracies}")
        text_file.write(f"\nMacro avg bleus: {macro_avg_bleus}")
        text_file.write(f"\nAll bleus: {all_bleus}")
        text_file.write(f"\nMacro avg rogues: {macro_avg_rogues}")
        text_file.write(f"\nAll rogues: {all_rogues}")
        text_file.write(f"\nMacro similarities: {macro_similarity}")
        text_file.write(f"\nAll similarities: {all_similarities}")
    return

def save_results_dimension(model, macro_avg_accuracy, direct_dim_avg, scale_bar_avg, all_accuracies, macro_avg_bleus,
                         all_bleus, macro_avg_rogues, all_rogues, macro_similarity, all_similarities):
    print(f"Model: {model}")
    print(f"\nMacro avg: {macro_avg_accuracy}")
    print(f"\nDirect Dimension avg: {direct_dim_avg}")
    print(f"\nScale Bar avg: {scale_bar_avg}")
    print(f"\nAll accuracies: {all_accuracies}")
    print(f"\nMacro avg bleus: {macro_avg_bleus}")
    print(f"\nAll bleus: {all_bleus}")
    print(f"\nMacro avg rogues: {macro_avg_rogues}")
    print(f"\nAll rogues: {all_rogues}")
    print(f"\nMacro avg similarities: {macro_similarity}")
    print(f"\nAll rogues: {all_similarities}")

    # Save results to txt file
    with open(f"dimension_{question_type}_evaluation_{model}.txt", "w") as text_file:
        text_file.write(f"Model: {model}")
        text_file.write(f"\nMacro avg: {macro_avg_accuracy}")
        text_file.write(f"\nDirect Dimension avg: {direct_dim_avg}")
        text_file.write(f"\nScale Bar avg: {scale_bar_avg}")
        text_file.write(f"\nAll accuracies: {all_accuracies}")
        text_file.write(f"\nMacro avg bleus: {macro_avg_bleus}")
        text_file.write(f"\nAll bleus: {all_bleus}")
        text_file.write(f"\nMacro avg rogues: {macro_avg_rogues}")
        text_file.write(f"\nAll rogues: {all_rogues}")
        text_file.write(f"\nMacro similarities: {macro_similarity}")
        text_file.write(f"\nAll similarities: {all_similarities}")
    return


if question_type == 'functional_performance':
    # Compute the accuracy of the responses
    macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues, macro_similarity, all_similarities = eval_functional_performance_qa(csv_name)

    # Print and save the results
    save_results_functional_performance(model, macro_avg_accuracy, all_accuracies, macro_avg_bleus, all_bleus, macro_avg_rogues, all_rogues, macro_similarity, all_similarities)

else:
    macro_avg_accuracy, direct_dim_avg, scale_bar_avg, all_accuracies, macro_avg_bleus, all_bleus, \
                macro_avg_rogues, all_rogues, macro_similarity, all_similarities = eval_dimensions_qa(csv_name)
    
    save_results_dimension(model, macro_avg_accuracy, direct_dim_avg, scale_bar_avg, all_accuracies, macro_avg_bleus,
                        all_bleus, macro_avg_rogues, all_rogues, macro_similarity, all_similarities)