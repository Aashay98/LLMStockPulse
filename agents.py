from typing import List

from langchain.agents import create_tool_calling_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate

from tools import (
    get_market_sentiment_news,
    get_news_from_newsapi,
    get_social_sentiment,
    get_stock_analysis,
    get_stock_data,
    process_search_tool,
    tavily_search,
)


def create_enhanced_prompt(
    system_message: str, include_context: bool = True
) -> ChatPromptTemplate:
    """Create enhanced prompt template with better context handling."""
    messages = [("system", system_message)]

    if include_context:
        messages.append(("placeholder", "{chat_history}"))

    messages.extend(
        [
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    return ChatPromptTemplate.from_messages(messages)


def create_stock_data_agent(llm):
    """Create specialized stock data agent with enhanced capabilities."""
    tools = [get_stock_data, get_stock_analysis]

    system_message = """You are a professional stock data analyst with expertise in financial markets.

Your responsibilities:
- Fetch and analyze real-time stock data including prices, volumes, and technical indicators
- Provide comprehensive fundamental analysis including P/E ratios, market cap, and financial metrics
- Generate actionable investment insights based on quantitative data
- Explain technical indicators in simple terms for users

Always provide specific, data-driven responses with proper context and disclaimers about investment risks."""

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)


def create_sentiment_agent(llm):
    """Create specialized sentiment analysis agent."""
    tools = [get_market_sentiment_news, get_news_from_newsapi]

    system_message = """You are a market sentiment specialist focused on news analysis and market psychology.

Your responsibilities:
- Analyze news sentiment and its potential market impact
- Identify trending topics and market narratives
- Assess the reliability and bias of news sources
- Correlate sentiment trends with potential price movements

Provide balanced analysis that considers both positive and negative sentiment factors."""

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)


def create_social_sentiment_agent(llm):
    """Create specialized agent for social media sentiment."""
    tools = [get_social_sentiment]

    system_message = """You analyze social media discussions on Reddit and Twitter to gauge market sentiment.

Your responsibilities:
- Fetch recent posts and comments about companies or topics
- Summarize positive and negative opinions
- Provide an overall sentiment score

Focus on capturing the crowd's mood from social platforms."""

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)


def create_insights_agent(llm):
    """Create specialized insights generation agent."""
    tools = [tavily_search, process_search_tool]

    system_message = """You are a financial research specialist who generates comprehensive market insights.

Your responsibilities:
- Conduct thorough research using web sources and financial databases
- Synthesize information from multiple sources to provide holistic views
- Identify market trends, opportunities, and risks
- Provide actionable insights for investment decision-making

Focus on delivering well-researched, balanced perspectives that help users make informed decisions."""

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)


def create_general_purpose_agent(llm):
    """Create general-purpose financial assistant."""
    tools = [tavily_search]

    system_message = """You are a knowledgeable financial assistant capable of handling diverse queries.

Your responsibilities:
- Answer general financial and investment questions
- Provide educational content about markets and investing
- Help users understand financial concepts and terminology
- Offer guidance on investment strategies and portfolio management

Always prioritize accuracy and provide educational value in your responses."""

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)
