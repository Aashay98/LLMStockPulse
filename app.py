import logging
import uuid
from datetime import datetime
from typing import Dict

import streamlit as st
import torch
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

import config
from agents import *
from constant import PAGE_TITLE, UI_CSS
from database import init_db
from error_tracking import capture_exception, init_sentry
from hitl import approve_hitl_response, reject_hitl_response
from log_config import configure_logging, set_correlation_id
from metrics import (
    ERROR_COUNTER,
    REQUEST_COUNTER,
    REQUEST_LATENCY,
    start_metrics_server,
)
from storage import append_history, load_history, load_relevant_history
from tools import get_price_chart
from tracing import get_tracer
from ui import display_sidebar, login_screen
from utils import (
    classify_query_chain,
    extract_ticker_symbol,
    friendly_error_message,
    generate_insights_prompt,
)

# Configure logging
configure_logging()
set_correlation_id(str(uuid.uuid4()))
init_sentry(config.SENTRY_DSN)
start_metrics_server(config.METRICS_PORT)
tracer = get_tracer(__name__)

logger = logging.getLogger(__name__)

init_db()

torch.classes.__path__ = []  # add this line to manually set it to empty.

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown(
    UI_CSS,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_llm():
    """Initialize and cache the LLM."""
    try:
        return ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
            api_key=config.GROQ_API_KEY,
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        st.stop()


@st.cache_resource
def initialize_agents(_llm):
    """Initialize and cache all agents."""
    try:
        agents = {
            "stock_data": create_stock_data_agent(_llm),
            "sentiment": create_sentiment_agent(_llm),
            "social_sentiment": create_social_sentiment_agent(_llm),
            "insights": create_insights_agent(_llm),
            "general": create_general_purpose_agent(_llm),
        }
        return agents
    except Exception as e:
        st.error(f"Failed to initialize agents: {e}")
        st.stop()


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "conversation_history": [],
        "latest_response": "",
        "last_user_query": "",
        "hitl_mode": False,
        "pending_hitl_response": None,
        "hitl_log": [],
        "error_count": 0,
        "regen_requested": False,
        "authenticated": False,
        "user_id": "",
        "current_conversation": None,
        "conversations": [],
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Load persistent conversation history if available
    if (
        not st.session_state["conversation_history"]
        and st.session_state.get("user_id")
        and st.session_state.get("current_conversation") is not None
    ):
        st.session_state["conversation_history"] = load_history(
            st.session_state["user_id"],
            st.session_state["current_conversation"],
        )

    # Initialize memories for each agent
    memory_keys = [
        "stock_memory",
        "sentiment_memory",
        "social_memory",
        "insights_memory",
        "general_memory",
    ]
    for memory_key in memory_keys:
        if memory_key not in st.session_state:
            st.session_state[memory_key] = ConversationBufferWindowMemory(
                k=config.MEMORY_WINDOW_SIZE,
                return_messages=True,
                memory_key="chat_history",
            )


def create_agent_executors(agents: Dict, memories: Dict) -> Dict[str, AgentExecutor]:
    """Create agent executors with proper tool assignments."""
    from tools import (
        get_market_sentiment_news,
        get_news_from_newsapi,
        get_social_sentiment,
        get_stock_analysis,
        get_stock_data,
        process_search_tool,
        tavily_search,
    )

    executors = {
        "stock_data": AgentExecutor(
            agent=agents["stock_data"],
            tools=[get_stock_data, get_stock_analysis],
            memory=memories["stock_memory"],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=config.MAX_AGENT_ITERATIONS,
        ),
        "sentiment": AgentExecutor(
            agent=agents["sentiment"],
            tools=[get_market_sentiment_news, get_news_from_newsapi],
            memory=memories["sentiment_memory"],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=config.MAX_AGENT_ITERATIONS,
        ),
        "social_sentiment": AgentExecutor(
            agent=agents["social_sentiment"],
            tools=[get_social_sentiment],
            memory=memories["social_memory"],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=config.MAX_AGENT_ITERATIONS,
        ),
        "insights": AgentExecutor(
            agent=agents["insights"],
            tools=[tavily_search, process_search_tool],
            memory=memories["insights_memory"],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=config.MAX_AGENT_ITERATIONS,
        ),
        "general": AgentExecutor(
            agent=agents["general"],
            tools=[tavily_search],
            memory=memories["general_memory"],
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=config.MAX_AGENT_ITERATIONS,
        ),
    }

    return executors


def execute_agent_safely(executor: AgentExecutor, input_data: Dict) -> Dict[str, str]:
    """Execute agent with proper error handling."""
    try:
        from rate_limit import check_api_usage

        check_api_usage("groq")
        result = executor.invoke(input_data)
        return {"success": True, "output": result.get("output", "No output generated")}
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {"success": False, "error": friendly_error_message(str(e))}


def multi_agent_query(query: str) -> str:
    """Enhanced multi-agent query processing with tracing and metrics."""

    user_id = st.session_state.get("user_id", "anonymous")
    REQUEST_COUNTER.labels(user=user_id).inc()
    with tracer.start_as_current_span("multi_agent_query"), REQUEST_LATENCY.labels(
        user=user_id
    ).time():
        try:

            # Initialize components
            llm = initialize_llm()
            agents = initialize_agents(llm)

            memories = {
                "stock_memory": st.session_state.stock_memory,
                "sentiment_memory": st.session_state.sentiment_memory,
                "social_memory": st.session_state.social_memory,
                "insights_memory": st.session_state.insights_memory,
                "general_memory": st.session_state.general_memory,
            }

            executors = create_agent_executors(agents, memories)

            # Retrieve relevant context from persistent history
            user_id = st.session_state.get("user_id", "default")
            context_entries = load_relevant_history(
                user_id,
                st.session_state.get("current_conversation"),
                query,
                config.MEMORY_WINDOW_SIZE,
            )
            context_text = "\n".join(
                f"{e['role']}: {e['content']}" for e in context_entries
            )
            context_prefix = f"{context_text}\n\n" if context_text else ""

            # Classify query and determine which agents to use
            query_type = classify_query_chain(llm, query)
            logger.info(f"Query classified as: {query_type}")

            responses = []
            errors = []

            # Execute appropriate agents based on query type
            if query_type in ["stock", "both"]:
                with st.spinner("🔍 Analyzing stock data..."):
                    result = execute_agent_safely(
                        executors["stock_data"],
                        {
                            "chat_history": memories[
                                "stock_memory"
                            ].load_memory_variables({})["chat_history"],
                            "input": f"{context_prefix}Provide comprehensive stock analysis for: {query}",
                        },
                    )

                    if result["success"]:
                        responses.append(
                            f"## 📈 Stock Data Analysis\n{result['output']}"
                        )
                        memories["stock_memory"].save_context(
                            {"input": query}, {"output": result["output"]}
                        )
                        symbol = extract_ticker_symbol(query)
                        if symbol:
                            chart = get_price_chart(symbol)
                            st.plotly_chart(chart, use_container_width=True)
                    else:
                        errors.append(f"❌ Stock Analysis failed: {result['error']}")

            if query_type in ["sentiment", "both"]:
                with st.spinner("📰 Analyzing market sentiment..."):
                    result = execute_agent_safely(
                        executors["sentiment"],
                        {
                            "chat_history": memories[
                                "sentiment_memory"
                            ].load_memory_variables({})["chat_history"],
                            "input": f"{context_prefix}Analyze market sentiment and news for: {query}",
                        },
                    )

                    if result["success"]:
                        responses.append(f"## Sentiment Analysis\n{result['output']}")
                        memories["sentiment_memory"].save_context(
                            {"input": query}, {"output": result["output"]}
                        )
                    else:
                        errors.append(
                            f"❌ Sentiment Analysis failed: {result['error']}"
                        )

                with st.spinner("📣 Gathering social media sentiment..."):
                    result = execute_agent_safely(
                        executors["social_sentiment"],
                        {
                            "chat_history": memories[
                                "social_memory"
                            ].load_memory_variables({})["chat_history"],
                            "input": f"{context_prefix}Summarize social media sentiment for: {query}",
                        },
                    )

                    if result["success"]:
                        responses.append(f"## 🗣️ Social Sentiment\n{result['output']}")
                        memories["social_memory"].save_context(
                            {"input": query}, {"output": result["output"]}
                        )
                    else:
                        errors.append(f"❌ Social Sentiment failed: {result['error']}")

            # Always generate insights for comprehensive analysis
            if query_type in ["stock", "sentiment", "both", "general"]:
                with st.spinner("Generating insights..."):
                    insights_prompt = generate_insights_prompt(query, query_type)
                    result = execute_agent_safely(
                        executors["insights"],
                        {
                            "chat_history": memories[
                                "insights_memory"
                            ].load_memory_variables({})["chat_history"],
                            "input": f"{context_prefix}{insights_prompt}",
                        },
                    )

                    if result["success"]:
                        responses.append(f"## Market Insights\n{result['output']}")
                        memories["insights_memory"].save_context(
                            {"input": insights_prompt}, {"output": result["output"]}
                        )
                    else:
                        errors.append(
                            f"❌ Insights Generation failed: {result['error']}"
                        )

            # Combine responses
            if responses:
                final_response = "\n\n".join(responses)
            else:
                # Fallback to general agent if all specialized agents fail
                with st.spinner("🤖 Processing with general assistant..."):
                    result = execute_agent_safely(
                        executors["general"],
                        {
                            "chat_history": memories[
                                "general_memory"
                            ].load_memory_variables({})["chat_history"],
                            "input": f"{context_prefix}{query}",
                        },
                    )

                    if result["success"]:
                        final_response = f"## 🤖 General Analysis\n{result['output']}"
                    else:
                        final_response = "❌ I apologize, but I'm unable to process your request at the moment. Please try again later."

            # Append errors if any
            if errors:
                st.session_state.error_count += len(errors)
                ERROR_COUNTER.labels(user=user_id).inc(len(errors))
                final_response += f"\n\n## ⚠️ Warnings\n" + "\n".join(errors)

            return final_response

        except Exception as e:  # pragma: no cover - defensive
            ERROR_COUNTER.labels(user=user_id).inc()
            capture_exception(e)
            logger.error(f"Multi-agent query failed: {e}")
            st.session_state.error_count += 1
            return f"❌ An unexpected error occurred: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    if not st.session_state.authenticated:
        login_screen()
    if st.session_state.pop("regen_requested", False):
        try:
            new_resp = multi_agent_query(st.session_state.last_user_query)
        except Exception as e:
            logger.error(f"Regenerate failed: {e}")
            new_resp = f"❌ Regenerate failed: {e}"
        st.session_state.pending_hitl_response = new_resp
        st.session_state.hitl_edit_box = new_resp

    # Display header
    st.markdown(
        '<h1 class="main-header">Stock Insight Assistant</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Ask me about stock fundamentals, market sentiment, technical analysis, or general financial questions. "
        "I'm powered by multiple specialized AI agents! 🤖"
    )

    # Display sidebar
    display_sidebar()

    # Main chat interface
    col1, col2 = st.columns([3, 1])

    with col1:

        if st.session_state.conversation_history:
            for message in st.session_state.conversation_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input
        user_query = st.chat_input(
            "Ask about stock trends, company performance, or market insights...",
            key="main_chat_input",
        )

        # Process user input
        if user_query and user_query.strip():
            st.session_state.last_user_query = user_query

            # Display user message
            with st.chat_message("user"):
                st.write(user_query)

            # Generate response
            with st.chat_message("assistant"):
                generated_response = multi_agent_query(user_query)

                if st.session_state.hitl_mode:
                    # HITL mode - show draft for review
                    st.session_state.pending_hitl_response = generated_response
                    st.session_state.hitl_edit_box = generated_response

                    st.markdown("### 💡 Review Assistant's Draft Response")
                    st.info(
                        "Please review and edit the response below if needed, then click 'Approve' to add it to the conversation."
                    )

                    col_approve, col_regenerate, col_reject = st.columns([1, 1, 1])
                    with col_approve:
                        if st.button("✅ Approve & Send", type="primary"):
                            approve_hitl_response(user_query)
                            st.rerun()

                    with col_regenerate:
                        if st.button("🔄 Regenerate Response"):
                            st.session_state.regen_requested = True
                            new_response = multi_agent_query(
                                st.session_state.last_user_query
                            )
                            st.session_state.pending_hitl_response = new_response
                            st.session_state.hitl_edit_box = new_response
                            st.rerun()
                    with col_reject:
                        if st.button("🗑️ Reject"):
                            reject_hitl_response(user_query)
                            st.rerun()
                else:
                    # Direct mode - show response immediately
                    st.markdown(generated_response)
                    response_dict = [
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": generated_response},
                    ]
                    # Add to conversation history
                    st.session_state.conversation_history.extend(response_dict)
                    # Persist to storage
                    append_history(
                        response_dict,
                        st.session_state.get("user_id", "default"),
                        st.session_state.get("current_conversation"),
                    )
                    st.session_state.latest_response = generated_response

        elif st.session_state.hitl_mode and st.session_state.pending_hitl_response:
            # Show pending response (e.g., after regeneration)
            with st.chat_message("assistant"):
                st.markdown("### 💡 Review Assistant's Draft Response")
                st.info(
                    "Please review and edit the response below if needed, then click 'Approve' to add it to the conversation."
                )

                col_approve, col_regenerate, col_reject = st.columns([1, 1, 1])
                with col_approve:
                    if st.button("✅ Approve & Send", type="primary"):
                        approve_hitl_response(st.session_state.last_user_query)
                        st.rerun()

                with col_regenerate:
                    if st.button("🔄 Regenerate Response"):
                        st.session_state.regen_requested = True
                        st.rerun()

                with col_reject:
                    if st.button("🗑️ Reject"):
                        reject_hitl_response(st.session_state.last_user_query)
                        st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")
