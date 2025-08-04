from sre_constants import IN

from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from constant import (
    GENERAL_PURPOSE_AGENT_SYSTEM_MESSAGE,
    INSIGHT_AGENT_SYSTEM_MESSAGE,
    SENTIMENT_DATA_AGENT_SYSTEM_MESSAGE,
    SOCIAL_SENTIMENT_AGENT_SYSTEM_MESSAGE,
    STOCK_DATA_AGENT_SYSTEM_MESSAGE,
)
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

    system_message = STOCK_DATA_AGENT_SYSTEM_MESSAGE

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)


def create_sentiment_agent(llm):
    """Create specialized sentiment analysis agent."""
    tools = [get_market_sentiment_news, get_news_from_newsapi]

    system_message = SENTIMENT_DATA_AGENT_SYSTEM_MESSAGE

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)


def create_social_sentiment_agent(llm):
    """Create specialized agent for social media sentiment."""
    tools = [get_social_sentiment]

    system_message = SOCIAL_SENTIMENT_AGENT_SYSTEM_MESSAGE

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)


def create_insights_agent(llm):
    """Create specialized insights generation agent."""
    tools = [tavily_search, process_search_tool]

    system_message = INSIGHT_AGENT_SYSTEM_MESSAGE

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)


def create_general_purpose_agent(llm):
    """Create general-purpose financial assistant."""
    tools = [tavily_search]

    system_message = GENERAL_PURPOSE_AGENT_SYSTEM_MESSAGE

    prompt = create_enhanced_prompt(system_message)
    return create_tool_calling_agent(llm, tools, prompt)
