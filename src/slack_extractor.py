import datetime

import requests

from src.config.settings import SLACK_BOT_TOKEN, SLACK_CHANNEL_ID


def fetch_slack_messages():
    """
    Fetch messages posted in the given Slack channel within the last 24 hours
    (or however you’d like to filter).
    """
    url = "https://slack.com/api/conversations.history"
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    # For daily runs, let's fetch from 'now - 24 hours' to 'now'.
    now = datetime.datetime.utcnow()
    oldest = int((now - datetime.timedelta(days=1)).timestamp())
    latest = int(now.timestamp())

    params = {
        "channel": SLACK_CHANNEL_ID,
        "oldest": oldest,
        "latest": latest,
        "inclusive": True,
        "limit": 100,  # adjust if needed
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Slack API error: {response.text}")

    data = response.json()
    if not data.get("ok"):
        raise Exception(f"Slack API error: {data}")

    # Return list of messages as text
    messages = data.get("messages", [])
    return messages
