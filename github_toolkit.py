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

# Filter tools to get the "Get Issue" tool
tools = [tool for tool in toolkit.get_tools() if tool.name in ["Get Issue", "Get Issues"]]
assert len(tools) == 2
tools[0].name = "get_issue"
tools[1].name = "get_issues"

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