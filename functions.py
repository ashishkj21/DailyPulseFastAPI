import os
import json
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

# Load environment variables and set up GitHub credentials
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

def store_github_data(user_id, data):
    filename = f"{user_id}_github_data.json"
    with open(filename, "w") as f:
        json.dump(data, f)

def load_github_data(user_id):
    filename = f"{user_id}_github_data.json"
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def collect_standup_update(user_input, user_id, slack_token, is_first_interaction=True):
    name = get_slack_user_name(user_id, slack_token)
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful assistant that collects daily standup updates from the user.

        Before responding to the user, gather all relevant information regarding their work from GitHub, including:
        - Comments on issues and pull requests
        - Open and closed pull requests
        - Issues worked on or created

        Use this information to draft a daily update for the user, which includes:
        - Accomplishments since the last update (based on closed PRs, resolved issues, and comments)
        - Plans for today (based on open PRs, ongoing issues, or recent comments)
        - Any blockers or challenges (e.g., unresolved issues, pending reviews, or comments indicating challenges)

        Present the draft to the user and ask: "Is this correct?". Allow the user to edit the information if needed.

        Remember to store the collected GitHub data locally to avoid repeated queries in the future.
        """),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # Initialize GitHub API wrapper and toolkit
    github = GitHubAPIWrapper()
    toolkit = GitHubToolkit.from_github_api_wrapper(github)
    tools = toolkit.get_tools()

    # Filter tools to get the required tools
    required_tool_names = [
        "Get Issues", "Get Issue", "Comment on Issue", "List open pull requests (PRs)",
        "Get Pull Request", "Overview of files included in PR", "Create Pull Request",
        "List Pull Requests' Files", "Create File", "Read File", "Update File", "Delete File",
        "Overview of existing files in Main branch", "Overview of files in current working branch",
        "List branches in this repository", "Set active branch", "Create a new branch",
        "Get files from a directory", "Search issues and pull requests", "Search code",
        "Create review request"
    ]

    tools = [tool for tool in toolkit.get_tools() if tool.name in required_tool_names]

    # Optionally, rename tools if needed
    tool_name_mapping = {
        "Get Issue": "get_issue",
        "Get Issues": "get_issues",
        "Comment on Issue": "comment_on_issue",
        "List open pull requests (PRs)": "list_open_prs",
        "Get Pull Request": "get_pr",
        "Overview of files included in PR": "overview_files_in_pr",
        "Create Pull Request": "create_pr",
        "List Pull Requests' Files": "list_pr_files",
        "Create File": "create_file",
        "Read File": "read_file",
        "Update File": "update_file",
        "Delete File": "delete_file",
        "Overview of existing files in Main branch": "overview_files_main_branch",
        "Overview of files in current working branch": "overview_files_working_branch",
        "List branches in this repository": "list_branches",
        "Set active branch": "set_active_branch",
        "Create a new branch": "create_new_branch",
        "Get files from a directory": "get_files_from_directory",
        "Search issues and pull requests": "search_issues_prs",
        "Search code": "search_code",
        "Create review request": "create_review_request"
    }

    for tool in tools:
        if tool.name in tool_name_mapping:
            tool.name = tool_name_mapping[tool.name]

    agent = create_tool_calling_agent(llm=chat, prompt=prompt_template, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    # Load previously stored GitHub data
    github_data = load_github_data(user_id)

    if github_data:
        # Use stored data in the agent's context
        context = f"Previously stored GitHub data: {json.dumps(github_data)}"
        user_input = f"{context}\n\n{user_input}"

    response = agent_executor.invoke({"input": user_input, "name": name})

    # Extract GitHub data from the response and store it
    github_data = {
        "issues": response.get("issues", []),
        "pull_requests": response.get("pull_requests", []),
        "comments": response.get("comments", [])
    }
    store_github_data(user_id, github_data)

    response["text"] = response["output"]
    response["user_input"] = response["input"]

    return response
