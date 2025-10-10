import openai

from src.config.settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """\
You are a data extraction assistant. You will receive messages posted in the “#releases” channel from Slack. \
Your goal is to parse each post and return a structured JSON object containing:
1. date: The date of the post or approximate date from context.
2. event_name: A short label describing the release or update.
3. product_area: e.g. "p2p", "swaps", "transactions", "kyc", "aml", "wallet", "other"
4. teams_or_people: A list of teams/individuals mentioned.
5. key_updates: A short list of the primary changes or features.
6. impact: Any mention of user/business impact or performance gains.
7. additional_notes: Additional context.

Output only valid JSON without extra commentary. If info is not found, use null or empty strings.
"""


def parse_message_with_llm(message_text: str) -> dict:
    """
    Sends the Slack message to the LLM with a system prompt,
    expects a JSON response with the required fields.
    """
    try:
        # ChatCompletion example (OpenAI)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Here is the raw Slack message:\n{message_text}",
                },
            ],
            temperature=1.0,
        )
        content = response["choices"][0]["message"]["content"]
        # Attempt to parse the LLM's JSON
        parsed = eval_json_safely(
            content
        )  # Implement a safe JSON parse (below, we show a naive approach)
        return parsed
    except Exception as e:
        # Log or raise error
        print(f"Error parsing message with LLM: {e}")
        return {}


def eval_json_safely(text: str) -> dict:
    """
    Naive approach: attempts to parse JSON from text.
    In production, use a robust JSON parser with error handling.
    """
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
