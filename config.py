import os
from typing import Dict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_credentials(value: str) -> Dict[str, str]:
    """Parse USER_CREDENTIALS env variable into a dict."""
    creds: Dict[str, str] = {}
    for pair in value.split(","):
        if ":" in pair:
            user, pwd = pair.split(":", 1)
            creds[user.strip()] = pwd.strip()
    return creds


# API Keys with validation
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
REDDIT_BEARER_TOKEN = os.getenv("REDDIT_BEARER_TOKEN")
USER_CREDENTIALS = parse_credentials(os.getenv("USER_CREDENTIALS", "admin:admin"))


# Validate required API keys
REQUIRED_KEYS = {
    "ALPHA_VANTAGE_API_KEY": ALPHA_VANTAGE_API_KEY,
    "NEWS_API_KEY": NEWS_API_KEY,
    "TAVILY_API_KEY": TAVILY_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
}

missing_keys = [key for key, value in REQUIRED_KEYS.items() if not value]
if missing_keys:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_keys)}"
    )

# API Configuration
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
NEWS_API_BASE_URL = "https://newsapi.org/v2/everything"

# Rate limiting and caching settings
REQUEST_TIMEOUT = 25
MAX_RETRIES = 4
CACHE_TTL = 300  # 5 minutes
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS"))
# Number of recent messages to retain in each agent's memory window
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE"))
# SQLAlchemy database URL (e.g., postgresql+psycopg2://user:pass@host/db)
DB_URL = os.getenv("DB_URL")
