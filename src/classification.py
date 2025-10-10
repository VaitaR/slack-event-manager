def classify_product_area(message_text: str) -> str:
    """
    Fallback or additional classifier if the LLM doesn't give a product area.
    Scans text for keywords.
    """
    lowered = message_text.lower()
    if "p2p" in lowered:
        return "p2p"
    elif "swap" in lowered:
        return "swaps"
    elif "kyc" in lowered:
        return "kyc"
    elif "aml" in lowered:
        return "aml"
    elif "transaction" in lowered:
        return "transactions"
    elif "wallet" in lowered:
        return "wallet"
    # ... etc ...
    else:
        return "other"
