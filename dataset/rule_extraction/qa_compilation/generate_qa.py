import pandas as pd


if __name__ == '__main__':
    # import extracted set of rules
    rules_pd = pd.read_csv("../rules_pdfplumber1_clean1.csv", encoding='utf-8-sig')     # TODO change path to clean one

    # Hardcoded terms of interest for compilation task
    terms = "Aerodynamic, Tractie System, Shutdown System, Accelerator Pedal Position Sensor/APPS, Brake Pedal, " \
            "Suspension, Battery, Chassis, Primary Structure, Critical Fasteners, Envelopes, Tube/Tubing, " \
            "Material properties, External Items, Impact Attenuator, Accumulator, Firewall, Powertrain, Catch " \
            "Cans, Thermal Protection, Scatter Shields, Coolant, Butt Joints, Inertia Switch, Transponder, " \
            "Brake Over Travel Switch/BOTS, Wiring, Grounded Low Voltage/GLV, Grounding, Lighting"

    ground_truth = {}
    # find the rules that contain the terms
    for term in terms.split(", "):
        ground_truth[term] = []
        for subterm in term.split("/"):
            # Find the rules that contain the subterm
            relevant_rules = rules_pd[rules_pd['rule_text'].str.contains(subterm, case=False, na=False)]
            # add the subrules as well # TODO

            # add rules that are mentioned in the rule # TODO

            # add each rule_number in relevant_rules to the ground truth dictionary
            for index, row in relevant_rules.iterrows():
                ground_truth[term].append(row['rule_number'])

    # create the questions
    qa = []
    for term, ground_truth_rules in ground_truth.items():
        # create the question
        question = f"We are a student engineering team designing a vehicle for the FSAE competition. Attached is the " \
                   f"FSAE rules document. Please list all rules relevant to `{term}`. Answer with only the rule numbers " \
                   f"(i.e.: AA.1.1.1) separated by commas and no other words.\n\n" \
                   f"The rules relevant to `{term}` are:\n"

        qa.append([question, ground_truth_rules])

    # Export questions and answers to compilation.csv
    pd.DataFrame(qa, columns=['question', 'answer']).to_csv("data/compilation.csv", index=False)

    print(len(qa))
