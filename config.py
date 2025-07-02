import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# API Keys with validation
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
CACHE_TTL = 300  # 5 minutes
