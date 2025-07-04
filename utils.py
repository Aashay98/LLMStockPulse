import logging
import re
from typing import List

import tiktoken

from exceptions import ValidationException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_stock_symbol(symbol: str) -> str:
    """Validate and clean stock symbol."""
    if not symbol or not isinstance(symbol, str):
        raise ValidationException("Stock symbol must be a non-empty string")

    # Clean and validate symbol format
    symbol = symbol.strip().upper()
    if not re.match(r"^[A-Z]{1,5}$", symbol):
        raise ValidationException(f"Invalid stock symbol format: {symbol}")

    return symbol


def generate_suggestions_from_topic(topic: str) -> List[str]:
    """Generate contextual suggestions based on topic."""
    if not topic:
        return []

    topic = topic.lower()

    if any(
        keyword in topic
        for keyword in ["stock", "price", "market", "nasdaq", "dow", "s&p"]
    ):
        return [
            "Compare this stock with MSFT or AAPL",
            "What is the earnings trend for this stock?",
            "Show recent insider trading activity",
            "What's the long-term outlook?",
        ]
    elif any(keyword in topic for keyword in ["sentiment", "news", "opinion", "trend"]):
        return [
            "Summarize recent analyst opinions",
            "What's the social media sentiment?",
            "Any upcoming events that could shift sentiment?",
            "Compare this sentiment to last month",
        ]
    else:
        return [
            "Can you go deeper into this topic?",
            "What are the key risks involved?",
            "Give a beginner summary",
            "What are related stocks to watch?",
        ]


def generate_insights_prompt(query: str, query_type: str) -> str:
    """Generate context-aware prompt for insights agent."""
    prompts = {
        "stock": f"Generate a comprehensive financial analysis and investment insights for {query}. "
        f"Consider earnings reports, revenue trends, P/E ratio, market positioning, and competitive landscape.",
        "both": f"Provide integrated financial analysis and sentiment insights for {query}. "
        f"Combine technical analysis with market sentiment and news impact.",
        "sentiment": f"Analyze market sentiment and news impact for {query}. "
        f"Summarize key trends, opinions, and potential market implications.",
        "general": f"Provide detailed insights and comprehensive analysis for {query}. "
        f"Consider relevant facts, trends, and broader market context.",
    }

    return prompts.get(query_type, f"Provide insights and analysis for {query}.")


def classify_query(query: str) -> str:
    """Classify query type with improved keyword matching."""
    if not query:
        return "general"

    query_lower = query.lower()

    stock_keywords = [
        "stock",
        "market",
        "share",
        "nasdaq",
        "dow jones",
        "s&p",
        "finance",
        "investment",
        "trading",
        "equity",
        "portfolio",
        "earnings",
        "dividend",
    ]
    sentiment_keywords = [
        "sentiment",
        "news",
        "social media",
        "opinion",
        "trends",
        "buzz",
        "analyst",
        "rating",
        "recommendation",
    ]

    is_stock = any(keyword in query_lower for keyword in stock_keywords)
    is_sentiment = any(keyword in query_lower for keyword in sentiment_keywords)

    if is_stock and is_sentiment:
        return "both"
    elif is_stock:
        return "stock"
    elif is_sentiment:
        return "sentiment"
    else:
        return "general"


def trim_text_to_token_limit(
    text: str, max_tokens: int = 5900, encoding_name: str = "cl100k_base"
) -> str:
    """Trim text to token limit with proper error handling."""
    if not text:
        return ""

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)

        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = encoding.decode(tokens)
            logger.warning(f"Text trimmed from {len(tokens)} to {max_tokens} tokens")

        return text
    except Exception as e:
        logger.error(f"Error trimming text: {e}")
        # Fallback: simple character-based trimming
        return text[: max_tokens * 4] if len(text) > max_tokens * 4 else text


def clean_content(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""

    # Remove excessive whitespace and normalize
    text = re.sub(r"\s+", " ", text.strip())
    # Remove special characters that might cause issues
    text = re.sub(r"[^\w\s\-.,!?()$%]", "", text)

    return text


def safe_float_conversion(value: str, default: float = 0.0) -> float:
    """Safely convert string to float with fallback."""
    try:
        return float(value) if value and value != "None" else default
    except (ValueError, TypeError):
        return default


def format_large_number(number: float) -> str:
    """Format large numbers for better readability."""
    if number >= 1e12:
        return f"${number/1e12:.2f}T"
    elif number >= 1e9:
        return f"${number/1e9:.2f}B"
    elif number >= 1e6:
        return f"${number/1e6:.2f}M"
    elif number >= 1e3:
        return f"${number/1e3:.2f}K"
    else:
        return f"${number:.2f}"


def friendly_error_message(error: str) -> str:
    """Provide user-friendly messages for common errors."""
    if "max iterations" in error.lower():
        return (
            "The agent reached its internal reasoning limit. "
            "Please try rephrasing your request or provide more details."
        )
    return error
