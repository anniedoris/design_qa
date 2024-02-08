import pandas as pd

# Function to convert data into QA pairs
def create_qa_pairs(file_name):

    questions = []
    answers = []

    df = pd.read_csv('data/' + file_name + '.csv')
    
    for i, row in df.iterrows():
        question = "We are a student FSAE team designing a vehicle for this year's competition." \
            " Attached is the FSAE rule document. " + row['question']
        questions.append(question)
        answers.append(row['answer'])
    
    return questions, answers

# Example of how to call the function, returns list of questions and list of corresponding answers
questions, answers = create_qa_pairs('compilation')
