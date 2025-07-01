import asyncio

import faiss  # For fast similarity search
import numpy as np
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from config import ALPHA_VANTAGE_API_KEY, NEWS_API_KEY
from utils import clean_content, trim_text_to_token_limit


@tool("stock_api_tool", return_direct=False)
def get_stock_data(stock_symbol: str, data_type: str = "intraday") -> str:
    """
    Fetches stock data for a given stock symbol.

    Available data_type options:
    - "intraday": Latest stock price (1-minute interval)
    - "daily": Daily adjusted closing prices
    - "fundamental": Company overview (market cap, EPS, PE ratio)
    - "indicators": Technical indicators (RSI, MACD)
    """

    BASE_URL = "https://www.alphavantage.co/query"
    params = {"symbol": stock_symbol, "apikey": ALPHA_VANTAGE_API_KEY}

    if data_type == "intraday":
        params["function"] = "TIME_SERIES_INTRADAY"
        params["interval"] = "1min"
    elif data_type == "daily":
        params["function"] = "TIME_SERIES_DAILY_ADJUSTED"
    elif data_type == "fundamental":
        params["function"] = "OVERVIEW"
    elif data_type == "indicators":
        params["function"] = "RSI"
        params["interval"] = "daily"
        params["time_period"] = "14"
        params["series_type"] = "close"
    elif data_type == "financials":
        params["function"] = "INCOME_STATEMENT"
    else:
        return "Invalid data type. Choose from 'intraday', 'daily', 'fundamental', 'financials','market_sentiment_news', or 'indicators'."

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    try:
        if data_type == "intraday":
            latest_data = data["Time Series (1min)"]
            latest_timestamp = next(iter(latest_data))
            stock_info = latest_data[latest_timestamp]
            return f"Stock: {stock_symbol} - Open: {stock_info['1. open']}, High: {stock_info['2. high']}, Low: {stock_info['3. low']}, Close: {stock_info['4. close']}, Volume: {stock_info['5. volume']} at {latest_timestamp}"

        elif data_type == "daily":
            latest_data = data["Time Series (Daily)"]
            latest_date = next(iter(latest_data))
            stock_info = latest_data[latest_date]
            return f"Stock: {stock_symbol} - Open: {stock_info['1. open']}, High: {stock_info['2. high']}, Low: {stock_info['3. low']}, Close: {stock_info['4. close']}, Adjusted Close: {stock_info['5. adjusted close']}, Volume: {stock_info['6. volume']} on {latest_date}"

        elif data_type == "fundamental":
            return f"Company: {data['Name']} ({stock_symbol})\nMarket Cap: {data['MarketCapitalization']}\nEPS: {data['EPS']}\nPE Ratio: {data['PERatio']}\nDividend Yield: {data['DividendYield']}\nSector: {data['Sector']}"

        elif data_type == "indicators":
            rsi_data = data["Technical Analysis: RSI"]
            latest_date = next(iter(rsi_data))
            return f"Stock: {stock_symbol} - RSI: {rsi_data[latest_date]['RSI']} on {latest_date}"
        elif data_type == "financials":
            annual_reports = data["annualReports"][0]  # Latest financial year report
            return f"Company: {stock_symbol}\nRevenue: {annual_reports['totalRevenue']}\nNet Income: {annual_reports['netIncome']}\nProfit Margin: {annual_reports['grossProfit']}"

    except KeyError:
        return "Error fetching stock data. Check API limits or verify the stock symbol."


@tool("stock_news_api_tool", return_direct=False)
def get_market_sentiment_news(ticker: str = None, topics: str = None):
    """
    Fetches the latest market sentiment news for a given stock ticker or market topics.

    Parameters:
    - ticker (str, optional): Stock symbol (e.g., "AAPL") to fetch news for.
    - topics (str, optional): Market topics such as "Earnings", "IPO", etc.

    Returns:
    - str: Formatted string containing the top 3 news articles with title, source, sentiment, and URL.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    params = {"function": "NEWS_SENTIMENT", "apikey": ALPHA_VANTAGE_API_KEY}

    if ticker:
        params["tickers"] = ticker  # Fetch news specific to the stock
    if topics:
        params["topics"] = topics  # Filter by topics like Earnings, IPO, etc.

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    news_items = data.get("feed", [])
    if not news_items:
        return "No market news available."

    # Extract top 3 news articles
    news_summary = []
    for news in news_items[:3]:
        news_summary.append(
            f"**Title**: {news['title']}\n"
            f"**Source**: {news['source']}\n"
            f"**Sentiment**: {news['overall_sentiment_label']}\n"
            f"**URL**: {news['url']}\n"
        )

    return "\n".join(news_summary)


@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Fetches and extracts content from a given URL."""
    try:
        # Add headers to mimic a real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract text from specific tags to avoid noise
        text = " ".join(
            [p.get_text() for p in soup.find_all(["p", "h1", "h2", "h3", "article"])]
        )
        return text
    except requests.Timeout:
        return f"Timeout error while processing URL {url}"
    except Exception as e:
        return f"Error processing URL {url}: {str(e)}"


# Asynchronous function to process multiple URLs concurrently
async def process_multiple_urls(urls):
    loop = asyncio.get_event_loop()

    # Use partial to pass the function and its arguments
    tasks = [
        loop.run_in_executor(None, partial(process_search_tool, url)) for url in urls
    ]
    results = await asyncio.gather(*tasks)
    return results


# Function to fetch and return up to 5 search results from Tavily
@tool("tavily_search_tool", return_direct=False)
def tavily_search(query: str) -> list:
    """Enhanced search with RAG processing"""
    tavily = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        topic="news",
        days=30,
    )
    results = tavily.run(query)

    # Initialize Sentence-BERT Model for Embedding Generation
    sbert_model = SentenceTransformer("all-MiniLM-L12-v2")
    # Convert to LangChain Documents
    documents = [
        Document(
            page_content=trim_text_to_token_limit(res["content"]),
            metadata={"source": res["url"], "title": res.get("title", "")},
        )
        for res in results
    ]

    # Rest of your RAG processing
    search_contents = [clean_content(doc.page_content) for doc in documents]
    query_embedding = sbert_model.encode([query])[0]
    search_embeddings = sbert_model.encode(search_contents)

    index = faiss.IndexFlatL2(query_embedding.shape[0])
    index.add(np.array(search_embeddings))

    k = 3
    D, I = index.search(np.array([query_embedding]), k)

    relevant_docs = [documents[i] for i in I[0]]  # Return Document objects

    return relevant_docs


# NewsAPI tool to fetch news articles
@tool("news_api_tool", return_direct=False)
def get_news_from_newsapi(query: str) -> str:
    """
    Fetches the latest news articles from NewsAPI for a specific query.
    """
    BASE_URL = "https://newsapi.org/v2/everything"

    params = {
        "q": query,  # Use the query from Tavily search
        "apiKey": NEWS_API_KEY,
        "language": "en",  # You can adjust the language as needed
        "sortBy": "relevance",  # Sort by relevance or any other criteria
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    # Extract top 3 news articles
    if data.get("status") == "ok":
        articles = data.get("articles", [])
        if articles:
            news_summary = []
            for article in articles[:3]:  # Limit to top 3 articles
                news_summary.append(
                    f"**Title**: {article['title']}\n"
                    f"**Source**: {article['source']['name']}\n"
                    f"**Description**: {article['description']}\n"
                    f"**URL**: {article['url']}\n"
                )
            return "\n".join(news_summary)
    return "No news articles found."


@tool("get_stock_analysis_tool", return_direct=False)
def get_stock_analysis(symbol: str) -> str:
    """Return a brief buy/hold/sell analysis for ``symbol``.

    Args:
        symbol: Stock ticker symbol.
    """

    try:
        # Fetch stock overview (P/E ratio, market cap, dividend yield)
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        stock_data = requests.get(overview_url).json()

        if "Error Message" in stock_data or "Note" in stock_data:
            return f"❌ Error: Could not fetch data for {symbol}. API limit may be reached."

        pe_ratio = float(stock_data.get("PERatio", 0))
        market_cap = float(stock_data.get("MarketCapitalization", 0))
        dividend_yield = float(stock_data.get("DividendYield", 0))

        # Fetch RSI (Relative Strength Index)
        rsi_url = f"https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval=daily&time_period=14&series_type=close&apikey={ALPHA_VANTAGE_API_KEY}"
        rsi_data = requests.get(rsi_url).json()
        rsi_values = rsi_data.get("Technical Analysis: RSI", {})

        # Get latest available RSI value
        latest_rsi_date = next(iter(rsi_values), None)
        rsi_value = (
            float(rsi_values[latest_rsi_date]["RSI"]) if latest_rsi_date else 50
        )  # Default 50 if unavailable

        # Fetch latest stock price
        price_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        price_data = requests.get(price_url).json()
        stock_price = float(price_data.get("Global Quote", {}).get("05. price", 0))

        # Fetch News Sentiment
        news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        news_response = requests.get(news_url).json()
        news_sentiment = news_response.get("feed", [])

        # Compute average sentiment score from top 5 news articles
        total_sentiment = sum(
            news.get("overall_sentiment_score", 0) for news in news_sentiment[:5]
        )
        avg_sentiment = total_sentiment / max(
            len(news_sentiment[:5]), 1
        )  # Avoid division by zero

        # **Decision Logic**
        if rsi_value < 30 and pe_ratio < 20 and avg_sentiment > 0:
            recommendation = (
                "🔵 BUY: The stock is undervalued and news sentiment is positive."
            )
        elif rsi_value > 70 and avg_sentiment < 0:
            recommendation = (
                "🔴 SELL: The stock is overbought and news sentiment is negative."
            )
        else:
            recommendation = "🟡 HOLD: Market conditions are stable."

        return f"""
        📈 **Stock Analysis for {symbol}**
        - **Current Price:** ${stock_price:.2f}
        - **P/E Ratio:** {pe_ratio:.2f}
        - **Market Cap:** ${market_cap:,.0f}
        - **RSI (14-day):** {rsi_value:.2f}
        - **Dividend Yield:** {dividend_yield:.2%}
        - **News Sentiment Score:** {avg_sentiment:.2f}

        **Recommendation: {recommendation}**
        """

    except Exception as e:
        return f"❌ Error fetching data: {str(e)}"
