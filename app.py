import os
from fastapi import FastAPI, Request, HTTPException
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from functions import collect_standup_update

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# Initialize the Slack app
slack_app = App(token=SLACK_BOT_TOKEN)

# Initialize the FastAPI app
app = FastAPI()
handler = SlackRequestHandler(slack_app)


def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        slack_client = WebClient(token=SLACK_BOT_TOKEN)
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")


@slack_app.event("app_mention")
def handle_mentions(event, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        event (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    user_id = event["user"]
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    text = event["text"]

    response = collect_standup_update(text, user_id, slack_token)
    say(response)


@app.post("/slack/events")
async def slack_events(request: Request):
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    try:
        return await handler.handle(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
# uvicorn app:app --reload --port 3000

