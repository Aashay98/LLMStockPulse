from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from app import llm
from tools import (get_market_sentiment_news, get_news_from_newsapi,
                   get_stock_analysis, get_stock_data, process_search_tool,
                   tavily_search)


# Define specialized agents
def create_stock_data_agent(llm):
    tools = [get_stock_data, get_stock_analysis]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a stock data expert. Fetch and analyze stock data."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # Add agent_scratchpad placeholder
    ])
    return create_tool_calling_agent(llm, tools, prompt)

def create_sentiment_agent(llm):
    tools = [get_market_sentiment_news, get_news_from_newsapi]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a sentiment analysis expert. Analyze news sentiment."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # Add agent_scratchpad placeholder
    ])
    return create_tool_calling_agent(llm, tools, prompt)

def create_insights_agent(llm):
    tools = [tavily_search, process_search_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an insights generator. Provide detailed insights."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # Add agent_scratchpad placeholder
    ])
    return create_tool_calling_agent(llm, tools, prompt)

def create_general_purpose_agent(llm):
    tools = [tavily_search]  # General-purpose tool like Tavily
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a general-purpose assistant. Answer any query comprehensively."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return create_tool_calling_agent(llm, tools, prompt)

# Coordinator agent - Currently not used
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
