import difflib
import logging
import re
from enum import Enum
from typing import List

import tiktoken
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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


class QueryType(str, Enum):
    """Supported query categories."""

    STOCK = "stock"
    SENTIMENT = "sentiment"
    BOTH = "both"
    GENERAL = "general"


class QueryClassification(BaseModel):
    """Structured output for query classification."""

    query_type: QueryType = Field(
        description="Category of the user's request: stock, sentiment, both, or general"
    )


def classify_query_chain(llm: BaseLanguageModel, query: str) -> str:
    """Classify query using an LLM-powered chain with structured output."""
    if not query:
        return QueryType.GENERAL.value

    parser = PydanticOutputParser(pydantic_object=QueryClassification)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user's financial query into one of: stock, sentiment, both, or general. "
                "Return JSON following the given format instructions.\n{format_instructions}",
            ),
            ("human", "{query}"),
        ]
    )

    chain = prompt | llm | parser

    try:
        result: QueryClassification = chain.invoke(
            {
                "query": query,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return result.query_type.value
    except Exception as e:
        logger.warning(f"LLM classification failed ({e}); falling back to heuristics")
        return classify_query(query)


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


def diff_text(original: str, edited: str) -> str:
    """Return a unified diff between two text blocks."""
    if original is None:
        original = ""
    if edited is None:
        edited = ""
    diff = difflib.unified_diff(
        original.splitlines(),
        edited.splitlines(),
        fromfile="original",
        tofile="edited",
        lineterm="",
    )
    return "\n".join(diff)


def extract_ticker_symbol(text: str) -> str | None:
    """Extract the first likely ticker symbol from text."""
    if not text:
        return None
    match = re.search(r"\b[A-Z]{1,5}\b", text.upper())

    # First try to find a symbol preceded by a dollar sign (e.g. ``$AAPL`` or ``$msft``)
    match = re.search(r"\$([a-zA-Z]{1,5})", text)
    if match:
        return match.group(1).upper()

    # Otherwise look for standalone uppercase tokens like ``AAPL``
    match = re.search(r"\b[A-Z]{1,5}\b", text)
    return match.group(0) if match else None


def friendly_error_message(error: str) -> str:
    """Provide user-friendly messages for common errors."""
    if "max iterations" in error.lower():
        return (
            "The agent reached its internal reasoning limit. "
            "Please try rephrasing your request or provide more details."
        )
    return error


def validate_username(username: str) -> str:
    """Validate and clean username input."""
    if not username or not isinstance(username, str):
        raise ValidationException("Username must be a non-empty string")

    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        raise ValidationException(
            "Username must be 3-30 characters and contain only letters, numbers, and underscores"
        )

    return username


def validate_password(password: str) -> str:
    """Validate password strength."""
    if not password or not isinstance(password, str):
        raise ValidationException("Password must be a non-empty string")

    if (
        len(password) < 8
        or not re.search(r"[A-Za-z]", password)
        or not re.search(r"\d", password)
    ):
        raise ValidationException(
            "Password must be at least 8 characters long and include letters and numbers"
        )

    return password
