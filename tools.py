import asyncio
import logging
import time
from functools import partial
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from config import (
    ALPHA_VANTAGE_API_KEY,
    ALPHA_VANTAGE_BASE_URL,
    MAX_RETRIES,
    NEWS_API_BASE_URL,
    NEWS_API_KEY,
    REQUEST_TIMEOUT,
)
from exceptions import APIException
from utils import (
    clean_content,
    format_large_number,
    safe_float_conversion,
    trim_text_to_token_limit,
    validate_stock_symbol,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize sentence transformer model (cached)
@st.cache_resource
def get_sentence_transformer():
    """Load and cache the sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L12-v2")


def make_api_request(
    url: str, params: Dict[str, Any], timeout: int = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """Make API request with retry logic and error handling."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            # Check for API-specific error messages
            if "Error Message" in data:
                raise APIException(f"API Error: {data['Error Message']}")
            if "Note" in data:
                raise APIException(f"API Limit: {data['Note']}")

            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise APIException(
                    f"Failed to fetch data after {MAX_RETRIES} attempts: {e}"
                )
            time.sleep(2**attempt)  # Exponential backoff


@tool("stock_api_tool", return_direct=False)
def get_stock_data(stock_symbol: str, data_type: str = "intraday") -> str:
    """
    Fetch stock data with improved error handling and validation.

    Args:
        stock_symbol: Stock ticker symbol
        data_type: Type of data to fetch (intraday, daily, fundamental, indicators, financials)
    """
    try:
        # Validate stock symbol
        stock_symbol = validate_stock_symbol(stock_symbol)

        params = {"symbol": stock_symbol, "apikey": ALPHA_VANTAGE_API_KEY}

        # Configure API parameters based on data type
        data_type_configs = {
            "intraday": {"function": "TIME_SERIES_INTRADAY", "interval": "1min"},
            "daily": {"function": "TIME_SERIES_DAILY_ADJUSTED"},
            "fundamental": {"function": "OVERVIEW"},
            "indicators": {
                "function": "RSI",
                "interval": "daily",
                "time_period": "14",
                "series_type": "close",
            },
            "financials": {"function": "INCOME_STATEMENT"},
        }

        if data_type not in data_type_configs:
            return f"Invalid data type '{data_type}'. Choose from: {', '.join(data_type_configs.keys())}"

        params.update(data_type_configs[data_type])
        data = make_api_request(ALPHA_VANTAGE_BASE_URL, params)

        return _format_stock_data(data, stock_symbol, data_type)

    except Exception as e:
        logger.error(f"Error fetching stock data for {stock_symbol}: {e}")
        return f"❌ Error fetching stock data: {str(e)}"


def _format_stock_data(data: Dict[str, Any], symbol: str, data_type: str) -> str:
    """Format stock data based on type."""
    try:
        if data_type == "intraday":
            time_series = data.get("Time Series (1min)", {})
            if not time_series:
                return f"No intraday data available for {symbol}"

            latest_timestamp = next(iter(time_series))
            stock_info = time_series[latest_timestamp]

            return (
                f"📈 **{symbol} - Latest Intraday Data ({latest_timestamp})**\n"
                f"• Open: ${safe_float_conversion(stock_info.get('1. open')):.2f}\n"
                f"• High: ${safe_float_conversion(stock_info.get('2. high')):.2f}\n"
                f"• Low: ${safe_float_conversion(stock_info.get('3. low')):.2f}\n"
                f"• Close: ${safe_float_conversion(stock_info.get('4. close')):.2f}\n"
                f"• Volume: {int(safe_float_conversion(stock_info.get('5. volume'))):,}"
            )

        elif data_type == "daily":
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                return f"No daily data available for {symbol}"

            latest_date = next(iter(time_series))
            stock_info = time_series[latest_date]

            return (
                f"📊 **{symbol} - Daily Data ({latest_date})**\n"
                f"• Open: ${safe_float_conversion(stock_info.get('1. open')):.2f}\n"
                f"• High: ${safe_float_conversion(stock_info.get('2. high')):.2f}\n"
                f"• Low: ${safe_float_conversion(stock_info.get('3. low')):.2f}\n"
                f"• Close: ${safe_float_conversion(stock_info.get('4. close')):.2f}\n"
                f"• Adjusted Close: ${safe_float_conversion(stock_info.get('5. adjusted close')):.2f}\n"
                f"• Volume: {int(safe_float_conversion(stock_info.get('6. volume'))):,}"
            )

        elif data_type == "fundamental":
            return (
                f"🏢 **{data.get('Name', 'N/A')} ({symbol}) - Company Overview**\n"
                f"• Market Cap: {format_large_number(safe_float_conversion(data.get('MarketCapitalization')))}\n"
                f"• EPS: ${safe_float_conversion(data.get('EPS')):.2f}\n"
                f"• P/E Ratio: {safe_float_conversion(data.get('PERatio')):.2f}\n"
                f"• Dividend Yield: {safe_float_conversion(data.get('DividendYield')) * 100:.2f}%\n"
                f"• Sector: {data.get('Sector', 'N/A')}\n"
                f"• Industry: {data.get('Industry', 'N/A')}"
            )

        elif data_type == "indicators":
            rsi_data = data.get("Technical Analysis: RSI", {})
            if not rsi_data:
                return f"No RSI data available for {symbol}"

            latest_date = next(iter(rsi_data))
            rsi_value = safe_float_conversion(rsi_data[latest_date].get("RSI"))

            # Interpret RSI
            if rsi_value > 70:
                interpretation = "Overbought (Consider selling)"
            elif rsi_value < 30:
                interpretation = "Oversold (Consider buying)"
            else:
                interpretation = "Neutral"

            return (
                f"📈 **{symbol} - RSI Analysis ({latest_date})**\n"
                f"• RSI (14-day): {rsi_value:.2f}\n"
                f"• Interpretation: {interpretation}"
            )

        elif data_type == "financials":
            annual_reports = data.get("annualReports", [])
            if not annual_reports:
                return f"No financial data available for {symbol}"

            latest_report = annual_reports[0]
            return (
                f"💰 **{symbol} - Latest Annual Financials**\n"
                f"• Total Revenue: {format_large_number(safe_float_conversion(latest_report.get('totalRevenue')))}\n"
                f"• Net Income: {format_large_number(safe_float_conversion(latest_report.get('netIncome')))}\n"
                f"• Gross Profit: {format_large_number(safe_float_conversion(latest_report.get('grossProfit')))}\n"
                f"• Operating Income: {format_large_number(safe_float_conversion(latest_report.get('operatingIncome')))}"
            )

    except Exception as e:
        logger.error(f"Error formatting stock data: {e}")
        return f"Error formatting data for {symbol}: {str(e)}"


@tool("stock_news_api_tool", return_direct=False)
def get_market_sentiment_news(ticker: str = None, topics: str = None) -> str:
    """Fetch market sentiment news with improved error handling."""
    try:
        if ticker:
            ticker = validate_stock_symbol(ticker)

        params = {"function": "NEWS_SENTIMENT", "apikey": ALPHA_VANTAGE_API_KEY}

        if ticker:
            params["tickers"] = ticker
        if topics:
            params["topics"] = topics

        data = make_api_request(ALPHA_VANTAGE_BASE_URL, params)
        news_items = data.get("feed", [])

        if not news_items:
            return "No market news available at this time."

        # Format top 3 news articles with better structure
        news_summary = []
        for i, news in enumerate(news_items[:3], 1):
            sentiment_score = safe_float_conversion(
                news.get("overall_sentiment_score", 0)
            )
            sentiment_emoji = (
                "🟢"
                if sentiment_score > 0.1
                else "🔴" if sentiment_score < -0.1 else "🟡"
            )

            news_summary.append(
                f"**{i}. {news.get('title', 'No title')}**\n"
                f"📰 Source: {news.get('source', 'Unknown')}\n"
                f"{sentiment_emoji} Sentiment: {news.get('overall_sentiment_label', 'Neutral')} "
                f"({sentiment_score:.2f})\n"
                f"🔗 [Read more]({news.get('url', '#')})\n"
            )

        return f"📰 **Latest Market News**\n\n" + "\n".join(news_summary)

    except Exception as e:
        logger.error(f"Error fetching market sentiment news: {e}")
        return f"❌ Error fetching news: {str(e)}"


@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Process URL content with improved error handling."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text from relevant tags
        text_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "article", "div"])
        text = " ".join(
            [
                elem.get_text().strip()
                for elem in text_elements
                if elem.get_text().strip()
            ]
        )

        return clean_content(text)[:2000]  # Limit content length

    except requests.exceptions.Timeout:
        return f"⏱️ Timeout error while processing {url}"
    except requests.exceptions.RequestException as e:
        return f"🌐 Network error processing {url}: {str(e)}"
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return f"❌ Error processing {url}: {str(e)}"


@tool("tavily_search_tool", return_direct=False)
def tavily_search(query: str) -> List[Document]:
    """Enhanced search with RAG processing and better error handling."""
    try:
        tavily = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            topic="news",
            days=30,
        )

        results = tavily.run(query)
        if not results:
            return [
                Document(
                    page_content="No search results found.",
                    metadata={"source": "tavily"},
                )
            ]

        # Get sentence transformer model
        sbert_model = get_sentence_transformer()

        # Convert to documents
        documents = []
        for res in results:
            content = trim_text_to_token_limit(clean_content(res.get("content", "")))
            if content:  # Only add non-empty content
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": res.get("url", ""),
                            "title": res.get("title", ""),
                            "score": res.get("score", 0),
                        },
                    )
                )

        if not documents:
            return [
                Document(
                    page_content="No valid content found in search results.",
                    metadata={"source": "tavily"},
                )
            ]

        # RAG processing with similarity search
        search_contents = [doc.page_content for doc in documents]
        query_embedding = sbert_model.encode([query])[0]
        search_embeddings = sbert_model.encode(search_contents)

        # Create FAISS index
        index = faiss.IndexFlatL2(query_embedding.shape[0])
        index.add(np.array(search_embeddings))

        # Find most relevant documents
        k = min(3, len(documents))
        D, I = index.search(np.array([query_embedding]), k)

        relevant_docs = [documents[i] for i in I[0]]
        return relevant_docs

    except Exception as e:
        logger.error(f"Error in Tavily search: {e}")
        return [
            Document(
                page_content=f"Search error: {str(e)}",
                metadata={"source": "error", "error": str(e)},
            )
        ]


@tool("news_api_tool", return_direct=False)
def get_news_from_newsapi(query: str) -> str:
    """Fetch news from NewsAPI with improved formatting."""
    try:
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "relevance",
            "pageSize": 3,
        }

        data = make_api_request(NEWS_API_BASE_URL, params)

        if data.get("status") != "ok":
            return f"❌ NewsAPI error: {data.get('message', 'Unknown error')}"

        articles = data.get("articles", [])
        if not articles:
            return "No news articles found for your query."

        news_summary = []
        for i, article in enumerate(articles, 1):
            news_summary.append(
                f"**{i}. {article.get('title', 'No title')}**\n"
                f"📰 Source: {article.get('source', {}).get('name', 'Unknown')}\n"
                f"📝 {article.get('description', 'No description')}\n"
                f"🔗 [Read more]({article.get('url', '#')})\n"
            )

        return f"📰 **Latest News for '{query}'**\n\n" + "\n".join(news_summary)

    except Exception as e:
        logger.error(f"Error fetching news from NewsAPI: {e}")
        return f"❌ Error fetching news: {str(e)}"


@tool("get_stock_analysis_tool", return_direct=False)
def get_stock_analysis(symbol: str) -> str:
    """Comprehensive stock analysis with buy/hold/sell recommendation."""
    try:
        symbol = validate_stock_symbol(symbol)

        # Fetch multiple data points
        analysis_data = {}

        # Get overview data
        overview_params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        overview_data = make_api_request(ALPHA_VANTAGE_BASE_URL, overview_params)

        # Get current price
        quote_params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        quote_data = make_api_request(ALPHA_VANTAGE_BASE_URL, quote_params)

        # Get RSI
        rsi_params = {
            "function": "RSI",
            "symbol": symbol,
            "interval": "daily",
            "time_period": "14",
            "series_type": "close",
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        rsi_data = make_api_request(ALPHA_VANTAGE_BASE_URL, rsi_params)

        # Get news sentiment
        news_params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        news_data = make_api_request(ALPHA_VANTAGE_BASE_URL, news_params)

        # Extract key metrics
        pe_ratio = safe_float_conversion(overview_data.get("PERatio"))
        market_cap = safe_float_conversion(overview_data.get("MarketCapitalization"))
        dividend_yield = safe_float_conversion(overview_data.get("DividendYield"))

        current_price = safe_float_conversion(
            quote_data.get("Global Quote", {}).get("05. price")
        )

        # Get latest RSI
        rsi_values = rsi_data.get("Technical Analysis: RSI", {})
        latest_rsi_date = next(iter(rsi_values), None)
        rsi_value = (
            safe_float_conversion(rsi_values[latest_rsi_date]["RSI"])
            if latest_rsi_date
            else 50
        )

        # Calculate average sentiment
        news_items = news_data.get("feed", [])[:5]
        sentiment_scores = [
            safe_float_conversion(item.get("overall_sentiment_score", 0))
            for item in news_items
        ]
        avg_sentiment = (
            sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        )

        # Generate recommendation
        recommendation = _generate_recommendation(pe_ratio, rsi_value, avg_sentiment)

        return f"""
📈 **Comprehensive Analysis for {symbol}**

**📊 Key Metrics:**
• Current Price: ${current_price:.2f}
• Market Cap: {format_large_number(market_cap)}
• P/E Ratio: {pe_ratio:.2f}
• Dividend Yield: {dividend_yield * 100:.2f}%

**📈 Technical Indicators:**
• RSI (14-day): {rsi_value:.2f} {_interpret_rsi(rsi_value)}

**📰 Market Sentiment:**
• Average Sentiment Score: {avg_sentiment:.2f} {_interpret_sentiment(avg_sentiment)}
• Based on {len(news_items)} recent articles

**🎯 Investment Recommendation:**
{recommendation}

*Disclaimer: This analysis is for informational purposes only and should not be considered as financial advice.*
        """

    except Exception as e:
        logger.error(f"Error in stock analysis for {symbol}: {e}")
        return f"❌ Error analyzing {symbol}: {str(e)}"


def _generate_recommendation(
    pe_ratio: float, rsi_value: float, avg_sentiment: float
) -> str:
    """Generate investment recommendation based on metrics."""
    buy_signals = 0
    sell_signals = 0

    # P/E Ratio analysis
    if 0 < pe_ratio < 15:
        buy_signals += 1
    elif pe_ratio > 25:
        sell_signals += 1

    # RSI analysis
    if rsi_value < 30:
        buy_signals += 1
    elif rsi_value > 70:
        sell_signals += 1

    # Sentiment analysis
    if avg_sentiment > 0.1:
        buy_signals += 1
    elif avg_sentiment < -0.1:
        sell_signals += 1

    if buy_signals >= 2 and sell_signals == 0:
        return (
            "🟢 **BUY**: Multiple positive indicators suggest good buying opportunity"
        )
    elif sell_signals >= 2 and buy_signals == 0:
        return "🔴 **SELL**: Multiple negative indicators suggest selling pressure"
    elif buy_signals > sell_signals:
        return "🟡 **WEAK BUY**: Some positive signals, but proceed with caution"
    elif sell_signals > buy_signals:
        return "🟠 **WEAK SELL**: Some negative signals, consider reducing position"
    else:
        return "⚪ **HOLD**: Mixed signals suggest maintaining current position"


def _interpret_rsi(rsi_value: float) -> str:
    """Interpret RSI value."""
    if rsi_value > 70:
        return "(Overbought)"
    elif rsi_value < 30:
        return "(Oversold)"
    else:
        return "(Neutral)"


def _interpret_sentiment(sentiment: float) -> str:
    """Interpret sentiment score."""
    if sentiment > 0.1:
        return "(Positive)"
    elif sentiment < -0.1:
        return "(Negative)"
    else:
        return "(Neutral)"
