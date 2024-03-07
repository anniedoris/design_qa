import re
import pandas as pd
import sys
sys.path.append("..")
from common_prompts import prompt_preamble


def find_subrules(rule_num, all_rules):
    """
    Recursively find all subrules for a given rule number.

    :param rule_num: The rule number to find subrules for.
    :param all_rules: DataFrame containing all rules.
    :return: DataFrame of all subrules for the given rule number.
    """
    # Find child rules by checking if the rule_num is a prefix of other rule_nums
    subrules = all_rules[all_rules['rule_num'].str.startswith(rule_num + ".")]

    # Recursively find subrules for each child rule
    for _, subrule in subrules.iterrows():
        subrules = pd.concat([subrules, find_subrules(subrule['rule_num'], all_rules)])

    return subrules.drop_duplicates()  # Remove duplicates if any


if __name__ == '__main__':
    # import extracted set of rules
    rules_pd = pd.read_csv("../../dataset/docs/csv_rules/all_rules_extracted.csv", encoding='utf-8-sig')

    # Hardcoded terms of interest for compilation task
    terms = "Aerodynamic/Aerodynamics, Tractive System, Shutdown System, Accelerator Pedal Position Sensor/APPS, Brake Pedal, " \
            "Suspension, Battery, Chassis, Primary Structure, Critical Fasteners/Critical Fastener, Envelope, Tube/Tubing/Tubes, " \
            "Material properties/material/materials, External Items/External Item, Impact Attenuator, Accumulator, Firewall, Powertrain, Catch " \
            "Cans, Thermal Protection, Scatter Shields, Coolant, Butt Joints/Butt Joint, Inertia Switch, Transponder, " \
            "Brake Over Travel Switch/BOTS, Wiring, Grounded Low Voltage/GLV, Grounding, Lighting/Light/Lights"

    ground_truth = {}
    # find the rules that contain the terms
    for term in terms.split(", "):
        ground_truth[term] = []
        for subterm in term.split("/"):
            # Find the rules that contain the subterm
            relevant_rules = rules_pd[rules_pd['rule_text'].str.contains(subterm, case=False, na=False)]
            # add the subrules as well
            for index, row in relevant_rules.iterrows():
                subrules = find_subrules(row['rule_num'], rules_pd)
                relevant_rules = pd.concat([relevant_rules, subrules])

            # add rules that are mentioned in the rule
            for index, row in relevant_rules.iterrows():
                # regex search for rule number
                matches = re.findall(r'([A-Z]+\.\d+(\.\d+){1,2})', row['rule_text'])

                if matches:
                    for match in matches:
                        # append the rule and text to relevant_rules
                        rule_number = match[0]
                        relevant_rules = relevant_rules._append({'rule_num': rule_number}, ignore_index=True)

            # drop duplicated based on rule_num
            relevant_rules = relevant_rules.drop_duplicates(subset='rule_num')

            # add each rule_number in relevant_rules to the ground truth dictionary
            for index, row in relevant_rules.iterrows():
                ground_truth[term].append(row['rule_num'])

    # create the questions
    qa = []
    for term, ground_truth_rules in ground_truth.items():
        # create the question
        question = prompt_preamble + f"Please list all rules relevant to `{term}`. Answer with only the rule numbers " \
                   f"(i.e.: AA.1.1.1) separated by commas and no other words.\n\n" \
                   f"The rules relevant to `{term}` are:\n"

        qa.append([question, ground_truth_rules])

    # Export questions and answers to compilation.csv
    pd.DataFrame(qa, columns=['question', 'ground_truth']).to_csv("../../dataset/rule_extraction/rule_compilation_qa.csv", index=False)

    print(len(qa))
