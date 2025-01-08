import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
from dotenv import find_dotenv, load_dotenv
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())

os.environ["GITHUB_APP_ID"] = os.getenv("GITHUB_APP_ID")
os.environ["GITHUB_APP_PRIVATE_KEY"] = os.getenv("GITHUB_APP_PRIVATE_KEY")
os.environ["GITHUB_REPOSITORY"] = os.getenv("GITHUB_REPOSITORY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN")

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
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful assistant that collects daily standup updates from the user.
        
        Your goal is to help the user quickly provide their standup update, which includes:
        - Accomplishments since the last standup
        - Plans for today
        - Any blockers or challenges currently faced
        
        If the user's response is vague or unclear, ask smart follow-up questions to get more details.
        Proactively identify potential blockers from responses (e.g., if the user mentions "waiting for review" or "need input from team").
        Remember the user's preferred writing style (bullet points vs. paragraphs) for the update.
        
        Start your reply by saying: "Hi {name}, please provide your standup update:".
        """),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),  # Add this line
    ])

    # Initialize GitHub API wrapper and toolkit
    github = GitHubAPIWrapper()
    toolkit = GitHubToolkit.from_github_api_wrapper(github)
    tools = toolkit.get_tools()

    # defining tools
    tools = [tool for tool in toolkit.get_tools() if tool.name in ["Get Issue", "Get Issues"]]
    assert len(tools) == 2
    tools[0].name = "get_issue"
    tools[1].name = "get_issues"

    agent = create_tool_calling_agent(llm=chat, prompt=prompt_template, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"input": user_input, "name": name})
    response["text"] = response["output"]
    response["user_input"] = response["input"]

    return response