import getpass
import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ["GITHUB_APP_ID"] = os.getenv("GITHUB_APP_ID")
os.environ["GITHUB_APP_PRIVATE_KEY"] = os.getenv("GITHUB_APP_PRIVATE_KEY")
os.environ["GITHUB_REPOSITORY"] = os.getenv("GITHUB_REPOSITORY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN")

# Initialize GitHub API wrapper and toolkit
github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)

# Get tools from the toolkit and print their names
tools = toolkit.get_tools()
for tool in tools:
    print(tool.name)

# Import ChatOpenAI and initialize the model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Test the model with a sample input
response = llm("Hello, how are you?")
print(response)

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

print(tools)
print(type(tools))

# Create the agent executor
agent_executor = create_react_agent(llm, tools)

# Example query to get issues
example_query = "What is issue number 1?"

try:
    events = agent_executor.stream(
        {"messages": [("user", example_query)]},
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()
except Exception as e:
    print(f"Error: {e}")
    # Additional debugging information
    print(f"Repository: {os.environ['GITHUB_REPOSITORY']}")
    print(f"GitHub Token: {os.environ['GITHUB_TOKEN'][:4]}...")  # Print partial token for security