import re
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage # import AIMessage

chat = ChatBedrock(
    model_id="anthropic.claude-instant-v1",
    model_kwargs={"temperature": 0.1},
)

messages = [
    SystemMessage(content="This is a chatbot that answers questions about AWS Cloud services only")
]

non_aws_providers = ['azure', 'google cloud', 'gcp', 'alibaba cloud', 'ibm cloud', 'oracle cloud']
def is_aws_related(prompt):
    aws_keywords = ['aws', 's3', 'ec2', 'lambda', 'cloudfront', 'dynamodb', 'iam', 'cloud']
    if any(non_aws in prompt.lower() for non_aws in non_aws_providers):
      return False
    return any(keyword in prompt.lower() for keyword in aws_keywords)

def invoke(prompt):
    if not is_aws_related(prompt):
        print("Sorry, I only answer questions about AWS Cloud.")
        return

    messages.append(HumanMessage(content=prompt))

    result = chat.invoke(messages)

    messages.append(AIMessage(content=result.content))

    print("Chatbot:", result.content)
    print("Total tokens used:", result.additional_kwargs["usage"]["total_tokens"])

def chat_conversation():
    print("Start the conversation. Type 'exit' to end.")

    while True:
        user_prompt = input("You: ")

        if user_prompt.lower() == 'exit':
            print("Ending the conversation.")
            break

        invoke(user_prompt)

chat_conversation()