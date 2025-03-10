### Import Necessary LangChain Components
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
import requests
from langchain.tools import tool
from bs4 import BeautifulSoup
import requests
import asyncio
import aiohttp
import os
from langchain_groq import ChatGroq
import getpass

import streamlit as st

st.write("Hello world")

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
    

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
    
# Initialize LangChain's ChatGroq Model
llm = ChatGroq(temperature=0.5)

#API_KEY = "QYBCUX9XUW8ESTIU35U2M531COX26A02"

API_KEY ="YL41PNDL63AAWZOI"

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
    params = {"symbol": stock_symbol, "apikey": API_KEY}

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
            latest_data = data['Time Series (1min)']
            latest_timestamp = next(iter(latest_data))
            stock_info = latest_data[latest_timestamp]
            return f"Stock: {stock_symbol} - Open: {stock_info['1. open']}, High: {stock_info['2. high']}, Low: {stock_info['3. low']}, Close: {stock_info['4. close']}, Volume: {stock_info['5. volume']} at {latest_timestamp}"
        
        elif data_type == "daily":
            latest_data = data['Time Series (Daily)']
            latest_date = next(iter(latest_data))
            stock_info = latest_data[latest_date]
            return f"Stock: {stock_symbol} - Open: {stock_info['1. open']}, High: {stock_info['2. high']}, Low: {stock_info['3. low']}, Close: {stock_info['4. close']}, Adjusted Close: {stock_info['5. adjusted close']}, Volume: {stock_info['6. volume']} on {latest_date}"
        
        elif data_type == "fundamental":
            return f"Company: {data['Name']} ({stock_symbol})\nMarket Cap: {data['MarketCapitalization']}\nEPS: {data['EPS']}\nPE Ratio: {data['PERatio']}\nDividend Yield: {data['DividendYield']}\nSector: {data['Sector']}"
        
        elif data_type == "indicators":
            rsi_data = data['Technical Analysis: RSI']
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

    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": API_KEY
    }

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
            "Referer": "https://www.google.com/"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract text from specific tags to avoid noise
        text = " ".join([p.get_text() for p in soup.find_all(["p", "h1", "h2", "h3", "article"])])
        return text
    except requests.Timeout:
        return f"Timeout error while processing URL {url}"
    except Exception as e:
        return f"Error processing URL {url}: {str(e)}"

# Asynchronous function to process multiple URLs concurrently
async def process_multiple_urls(urls):
    loop = asyncio.get_event_loop()
    
    # Use partial to pass the function and its arguments
    tasks = [loop.run_in_executor(None, partial(process_search_tool, url)) for url in urls]
    results = await asyncio.gather(*tasks)
    return results



# Function to fetch and return up to 5 search results from Tavily
@tool("tavily_search_tool", return_direct=False)
def tavily_search(query: str) -> list:
    """Fetches search results for a given query using Tavily."""
    tavily_search = TavilySearchResults(max_results=10, search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True)
    return tavily_search.run(query)

# NewsAPI tool to fetch news articles
@tool("news_api_tool", return_direct=False)
def get_news_from_newsapi(query: str) -> str:
    """
    Fetches the latest news articles from NewsAPI for a specific query.
    """
    API_KEY = "d2afe10169b44e628b2131aed04ac7e4"  # Add your NewsAPI key here
    BASE_URL = "https://newsapi.org/v2/everything"

    params = {
        "q": query,  # Use the query from Tavily search
        "apiKey": API_KEY,
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
def get_stock_analysis(query: str) -> str:
    """
    Fetches stock financial data, technical indicators, and news sentiment analysis 
    for a given stock symbol and provides a Buy/Hold/Sell recommendation.
    """
    
    try:
        # Fetch stock overview (P/E ratio, market cap, dividend yield)
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={API_KEY}"
        stock_data = requests.get(overview_url).json()

        if "Error Message" in stock_data or "Note" in stock_data:
            return f"❌ Error: Could not fetch data for {symbol}. API limit may be reached."

        pe_ratio = float(stock_data.get("PERatio", 0))
        market_cap = float(stock_data.get("MarketCapitalization", 0))
        dividend_yield = float(stock_data.get("DividendYield", 0))

        # Fetch RSI (Relative Strength Index)
        rsi_url = f"https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval=daily&time_period=14&series_type=close&apikey={API_KEY}"
        rsi_data = requests.get(rsi_url).json()
        rsi_values = rsi_data.get("Technical Analysis: RSI", {})

        # Get latest available RSI value
        latest_rsi_date = next(iter(rsi_values), None)
        rsi_value = float(rsi_values[latest_rsi_date]["RSI"]) if latest_rsi_date else 50  # Default 50 if unavailable

        # Fetch latest stock price
        price_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
        price_data = requests.get(price_url).json()
        stock_price = float(price_data.get("Global Quote", {}).get("05. price", 0))

        # Fetch News Sentiment
        news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={API_KEY}"
        news_response = requests.get(news_url).json()
        news_sentiment = news_response.get("feed", [])

        # Compute average sentiment score from top 5 news articles
        total_sentiment = sum(news.get("overall_sentiment_score", 0) for news in news_sentiment[:5])
        avg_sentiment = total_sentiment / max(len(news_sentiment[:5]), 1)  # Avoid division by zero

        # **Decision Logic**
        if rsi_value < 30 and pe_ratio < 20 and avg_sentiment > 0:
            recommendation = "🔵 BUY: The stock is undervalued and news sentiment is positive."
        elif rsi_value > 70 and avg_sentiment < 0:
            recommendation = "🔴 SELL: The stock is overbought and news sentiment is negative."
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

# Define specialized agents
def create_stock_data_agent(llm):
    tools = [get_stock_data, get_stock_analysis]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a stock data expert. Fetch and analyze stock data."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # Add agent_scratchpad placeholder
    ])
    return create_tool_calling_agent(llm, tools, prompt)

def create_sentiment_agent(llm):
    tools = [get_market_sentiment_news]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sentiment analysis expert. Analyze news sentiment."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # Add agent_scratchpad placeholder
    ])
    return create_tool_calling_agent(llm, tools, prompt)


def create_insights_agent(llm):
    tools = [tavily_search, process_search_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an insights generator. Provide detailed insights."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # Add agent_scratchpad placeholder
    ])
    return create_tool_calling_agent(llm, tools, prompt)

def create_general_purpose_agent(llm):
    tools = [tavily_search]  # General-purpose tool like Tavily
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a general-purpose assistant. Answer any query comprehensively."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return create_tool_calling_agent(llm, tools, prompt)


# Coordinator agent
def create_coordinator_agent(llm):
    tools = []  # No tools needed for the coordinator
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the coordinator. Manage interactions between agents and combine their responses."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # Add agent_scratchpad placeholder
    ])
    return create_tool_calling_agent(llm, tools, prompt)

# Initialize agents (replace `llm` with your actual LLM instance)
stock_data_agent = create_stock_data_agent(llm)
sentiment_agent = create_sentiment_agent(llm)
insights_agent = create_insights_agent(llm)
general_purpose_agent = create_general_purpose_agent(llm)
coordinator_agent = create_coordinator_agent(llm)

# Agent executors
stock_data_executor = AgentExecutor(agent=stock_data_agent, tools=[get_stock_data, get_stock_analysis], verbose=True)
sentiment_executor = AgentExecutor(agent=sentiment_agent, tools=[get_market_sentiment_news], verbose=True)
insights_executor = AgentExecutor(agent=insights_agent, tools=[tavily_search, process_search_tool], verbose=True)
general_purpose_executor = AgentExecutor(agent=general_purpose_agent, tools=[tavily_search], verbose=True)
coordinator_executor = AgentExecutor(agent=coordinator_agent, tools=[], verbose=True)


def classify_query(query):
    """Classifies the query type: stock/finance, sentiment, or general-purpose.
    Handles cases where the query contains keywords from multiple categories."""
    stock_keywords = ["stock", "market", "share", "nasdaq", "dow jones", "finance", "investment"]
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

        
def multi_agent_query(query):
    responses = []
    errors = []
    query_type = classify_query(query)
    print(query_type)


    # Fetch stock data if it's a stock-related query
    if query_type in ["stock", "both"]:
        try:
            stock_data_response = stock_data_executor.invoke(
                {"input": f"Retrieve the latest stock data and market trends for {query}. Provide key statistics, including open, high, low, close, and volume."}
            )
            responses.append(f"**Stock Data Analysis**:\n{stock_data_response['output']}")
        except Exception as e:
            errors.append(f"❌ Stock Data Agent failed: {str(e)}")

    # Fetch sentiment analysis if it's related to financial sentiment
    if query_type in ["sentiment", "both"]:
        try:
            sentiment_response = sentiment_executor.invoke(
                {"input": f"Analyze the market sentiment for {query}. Summarize the tone of recent news articles, social media discussions, and investor opinions."}
            )
            responses.append(f"**Sentiment Analysis**:\n{sentiment_response['output']}")
        except Exception as e:
            errors.append(f"❌ Sentiment Agent failed: {str(e)}")
            
    # Step 3: Generate insights (generic input)
    if query_type in ["stock", "sentiment", "both", "general"]:
        try:
            insights_prompt = generate_insights_prompt(query, query_type)
            insights_response = insights_executor.invoke({"input": insights_prompt})
            responses.append(f"**Insights**:\n{insights_response['output']}")
        except Exception as e:
            errors.append(f"❌ Insights Agent failed: {str(e)}")

    # Combine responses
    final_response = "\n\n".join(responses) if responses else "No data available."

    # Append errors if any agents failed
    if errors:
        final_response += "\n\n**Errors**:\n" + "\n".join(errors)

    return final_response


# Example
query = "Can you state which llm models are better than Chatgpt 4 and if there are any new models that are better than Chatgpt 4 then explain them?"
response = multi_agent_query(query)
print(response)
