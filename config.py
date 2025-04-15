import os

from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
