import tiktoken


def generate_suggestions_from_topic(topic):
    topic = topic.lower()
    suggestions = []

    if any(
        keyword in topic
        for keyword in ["stock", "price", "market", "nasdaq", "dow", "s&p"]
    ):
        suggestions = [
            "Compare this stock with MSFT or AAPL",
            "What is the earnings trend for this stock?",
            "Show recent insider trading activity",
            "What's the long-term outlook?",
        ]
    elif any(keyword in topic for keyword in ["sentiment", "news", "opinion", "trend"]):
        suggestions = [
            "Summarize recent analyst opinions",
            "What’s the social media sentiment?",
            "Any upcoming events that could shift sentiment?",
            "Compare this sentiment to last month",
        ]
    else:
        suggestions = [
            "Can you go deeper into this topic?",
            "What are the key risks involved?",
            "Give a beginner summary",
            "What are related stocks to watch?",
        ]

    return suggestions


def generate_insights_prompt(query, query_type):
    """Generates a context-aware prompt for the insights agent based on the query type."""
    if query_type in ["stock", "both"]:
        return f"Generate a financial analysis and investment insights for {query}. Consider earnings reports, revenue trends, P/E ratio, and market positioning."
    elif query_type == "sentiment":
        return f"Provide insights based on the sentiment analysis for {query}. Summarize key trends, opinions, and potential implications."
    elif query_type == "general":
        return f"Provide detailed insights and analysis for {query}. Consider relevant facts, trends, and context."
    else:
        return f"Provide insights and analysis for {query}."


def classify_query(query):
    """Classifies the query type: stock/finance, sentiment, or general-purpose.
    Handles cases where the query contains keywords from multiple categories."""
    stock_keywords = [
        "stock",
        "market",
        "share",
        "nasdaq",
        "dow jones",
        "finance",
        "investment",
    ]
    sentiment_keywords = ["sentiment", "news", "social media", "opinion", "trends"]

    is_stock = any(keyword in query.lower() for keyword in stock_keywords)
    is_sentiment = any(keyword in query.lower() for keyword in sentiment_keywords)

    if is_stock and is_sentiment:
        return "both"  # Handle cases where the query is relevant to both categories
    elif is_stock:
        return "stock"
    elif is_sentiment:
        return "sentiment"
    else:
        return "general"  # Default to Tavily for general queries


# Function to count tokens and trim text if it exceeds the limit
def trim_text_to_token_limit(text, max_tokens=5900, encoding_name="cl100k_base"):
    """
    Trims the text to ensure it does not exceed the specified token limit.

    Args:
        text (str): The input text.
        max_tokens (int): The maximum number of tokens allowed.
        encoding_name (str): The encoding model to use (e.g., "cl100k_base" for GPT-4).

    Returns:
        str: The trimmed text.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]  # Trim to the first `max_tokens` tokens
        text = encoding.decode(tokens)  # Convert back to text
    return text


def clean_content(text):
    text = text.strip().replace("\\n", " ")
    return " ".join(text.split())  # Remove excess whitespace
