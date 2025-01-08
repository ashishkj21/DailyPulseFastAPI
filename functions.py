from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def get_slack_user_name(user_id, slack_token):
    client = WebClient(token=slack_token)
    try:
        response = client.users_info(user=user_id)
        return response["user"]["real_name"]
    except SlackApiError as e:
        print(f"Error fetching user info: {e.response['error']}")
        return "User"


def collect_standup_update(user_input, user_id, slack_token, is_first_interaction=True):
    name = get_slack_user_name(user_id, slack_token)
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    if is_first_interaction:
        template = """
        You are a helpful assistant that collects daily standup updates from the user.
        
        Your goal is to help the user quickly provide their standup update, which includes:
        - Accomplishments since the last standup
        - Plans for today
        - Any blockers or challenges currently faced
        
        If the user's response is vague or unclear, ask smart follow-up questions to get more details.
        Proactively identify potential blockers from responses (e.g., if the user mentions "waiting for review" or "need input from team").
        Remember the user's preferred writing style (bullet points vs. paragraphs) for the update.
        
        Start your reply by saying: "Hi {name}, please provide your standup update:". And then proceed with the conversation.
        """
    else:
        template = """
        You are a helpful assistant that collects daily standup updates from the user.
        
        Your goal is to help the user quickly provide their standup update, which includes:
        - Accomplishments since the last standup
        - Plans for today
        - Any blockers or challenges currently faced
        
        If the user's response is vague or unclear, ask smart follow-up questions to get more details.
        Proactively identify potential blockers from responses (e.g., if the user mentions "waiting for review" or "need input from team").
        Remember the user's preferred writing style (bullet points vs. paragraphs) for the update.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "User's response: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run({"user_input": user_input, "name": name})

    return response
