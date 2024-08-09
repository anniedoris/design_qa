import pandas as pd

num_idealrag = 5

presence_df = pd.read_csv('rule_presence_qa_for_idealrag.csv')
definition_df_idealrag = pd.read_csv('../../dataset/rule_comprehension/definition_idealrag' + str(num_idealrag) + '.csv')

idealrags = []

# print(definition_df_idealrag)

for i, row in presence_df.iterrows():
    component_name = row['component']
    print(f"Looking for component: {component_name}")
    relevant_context = definition_df_idealrag[definition_df_idealrag["ground_truth"] == component_name]['question'].values[0].split('```')[1]
    print(f"Found context: {relevant_context}")
    idealrags.append(relevant_context)

# Combine the new IdealRAG with the question we are trying to ask
full_question_plus_ideal_rag = []
for i, rag in enumerate(idealrags):
    original_prompt = presence_df.iloc[i]['question']
    full = original_prompt[:80] + f"Below is context from the FSAE rule document which might or might not " \
                                f"be relevant for the question: \n\n```\n{rag}\n```\n\n" + original_prompt[117:]
    full_question_plus_ideal_rag.append(full)

# Add everything together, ground truth and new questions
presence_df['full_question'] = full_question_plus_ideal_rag

# Swap column names
col1 = 'question'
col2 = 'full_question'
presence_df = presence_df.rename(columns={col1: 'temp', col2: col1, 'temp': col2})
presence_df.to_csv('presence_idealrag' + str(num_idealrag) + '.csv')

print(presence_df)