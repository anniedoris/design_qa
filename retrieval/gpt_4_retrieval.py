from openai import OpenAI
from time import sleep
from itertools import cycle

# client = OpenAI(api_key=open_ai_key)

# file_name = 

# # Upload a file with an "assistants" purpose
# file = client.files.create(
#     file=open(file_name, "rb"),
#     purpose='assistants'
# )

# # Add the file to the assistant
# assistant = client.beta.assistants.create(
#     instructions="You are a customer support chatbot. Use your knowledge base to best respond to customer queries.",
#     model="gpt-4-1106-preview",
#     tools=[{"type": "retrieval"}],
#     file_ids=[file.id]
# )

# message = client.beta.threads.messages.create(
#     thread_id=thread.id,
#     role="user",
#     content="I can not find in the PDF manual how to turn off this device.",
#     file_ids=[file.id]
# )

# # # Function to ask chat gpt a question and get the result
# # def ask_chat_gpt(question):
# #     completion = client.chat.completions.create(
# #     model="gpt-4-1106-preview",
# #     messages=[
# #         {"role": "system", "content": "You are an expert on NASA technical standards, and you are helping to create exam questions that test engineers' knowledge of the standards."},
# #         {"role": "user", "content": question}
# #     ]
# #     )
# #     return completion.choices[0].message.content


def upload_file(file_path):
    # Upload a file with an "assistants" purpose
    upload_file = client.files.create(
    file=open(file_path, "rb"),
    purpose='assistants')

    return upload_file


def create_assistant(file):
    # Add the file to the assistant
    assistant = client.beta.assistants.create(
        instructions="You are a mechanical designer chatbot. Use your knowledge base to best respond to customer queries.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file.id])

    return assistant


def run_thread(question):
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,)
    status_complete = False
    for i in cycle(["|", "/", "-", "\\"]):
        print(f"Loading... {i}",end='\r') # '\r' clears the previous output
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id)
        status_complete = run.status == "completed"
        if status_complete:
            break
        sleep(0.5)

        # Retrieve the message object
        messages = client.beta.threads.messages.list(
            thread_id=thread.id)

    # Extract the message content
    message_content = messages.data[0].content[0].text
    annotations = message_content.annotations
    citations = []

    # Iterate over the annotations and add footnotes
    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        message_content.value = message_content.value.replace(annotation.text, f' [{index}]')

        # Gather citations based on annotation attributes
        if (file_citation := getattr(annotation, 'file_citation', None)):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
        elif (file_path := getattr(annotation, 'file_path', None)):
            cited_file = client.files.retrieve(file_path.file_id)
            citations.append(f'[{index}] Click <here> to download {cited_file.filename}')
            # Note: File download functionality not implemented above for brevity

    # Add footnotes to the end of the message before displaying to user
    message_content.value += '\n\n' + '\n'.join(citations)

    return message_content.value

if __name__ == '__main__':
    file_path = 'docs/FSAE_Rules_2024_V1 (2).pdf'
    file = upload_file(file_path)
    assistant = create_assistant(file)

    question = "We are a student engineering team designing a vehicle for the FSAE competition. Attached is the FSAE rules document. Based on the rules, tell me, verbatim, what rule section V.1.1 states."

    response = run_thread(question)
    print(response)