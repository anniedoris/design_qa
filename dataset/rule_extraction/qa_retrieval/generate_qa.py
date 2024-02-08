import pandas as pd

# Function to convert data into QA pairs
def create_qa_pairs(csv_files):

    questions = []
    answers = []

    for file in csv_files:
        df = pd.read_csv('data/' + file + '_final.csv')
        
        for i, row in df.iterrows():
            rule_num = row['rule_num']
            rule_text = row['rule_text']
            question = "We are a student FSAE team designing a vehicle for this year's competition." \
                " Attached is the FSAE rule document. Tell me verbatim, what does rule " + str(rule_num) + " state?"
            questions.append(question)
            answers.append(rule_text)
    
    return questions, answers

# Example of how to call the function, returns list of questions and list of corresponding answers
questions, answers = create_qa_pairs(['V'])

# for i, q in enumerate(questions):
#     print(q)