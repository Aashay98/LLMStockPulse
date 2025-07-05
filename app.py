import logging
from datetime import datetime
from typing import Dict

import streamlit as st
import torch
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

import config
from agents import *
from storage import append_history, clear_history, load_history
from utils import (
    classify_query,
    diff_text,
    friendly_error_message,
    generate_insights_prompt,
    trim_text_to_token_limit,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.classes.__path__ = []  # add this line to manually set it to empty.

# Page configuration
st.set_page_config(
    page_title="Stock Insight Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .agent-response {
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    /* Chat bubbles */
    .stChatMessage {
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
    }
    .stChatMessage.user {
        background-color: var(--primary-color);
        color: #ffffff;
        align-self: flex-end;
    }
    .stChatMessage.assistant {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        align-self: flex-start;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_llm():
    """Initialize and cache the LLM."""
    try:
        return ChatGroq(
            model="llama-3.3-70b-versatile", temperature=0, api_key=config.GROQ_API_KEY
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
            "coordinator": create_coordinator_agent(_llm),
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
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Load persistent conversation history if available
    if not st.session_state["conversation_history"]:
        st.session_state["conversation_history"] = load_history()

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
            st.session_state[memory_key] = ConversationBufferMemory(
                return_messages=True, memory_key="chat_history"
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
        result = executor.invoke(input_data)
        return {"success": True, "output": result.get("output", "No output generated")}
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {"success": False, "error": friendly_error_message(str(e))}


def multi_agent_query(query: str) -> str:
    """Enhanced multi-agent query processing with better error handling."""
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

        # Classify query and determine which agents to use
        query_type = classify_query(query)
        logger.info(f"Query classified as: {query_type}")

        responses = []
        errors = []

        # Execute appropriate agents based on query type
        if query_type in ["stock", "both"]:
            with st.spinner("🔍 Analyzing stock data..."):
                result = execute_agent_safely(
                    executors["stock_data"],
                    {
                        "chat_history": memories["stock_memory"].load_memory_variables(
                            {}
                        )["chat_history"],
                        "input": f"Provide comprehensive stock analysis for: {query}",
                    },
                )

                if result["success"]:
                    responses.append(f"## 📈 Stock Data Analysis\n{result['output']}")
                    memories["stock_memory"].save_context(
                        {"input": query}, {"output": result["output"]}
                    )
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
                        "input": f"Analyze market sentiment and news for: {query}",
                    },
                )

                if result["success"]:
                    responses.append(f"## � Sentiment Analysis\n{result['output']}")
                    memories["sentiment_memory"].save_context(
                        {"input": query}, {"output": result["output"]}
                    )
                else:
                    errors.append(f"❌ Sentiment Analysis failed: {result['error']}")

            with st.spinner("📣 Gathering social media sentiment..."):
                result = execute_agent_safely(
                    executors["social_sentiment"],
                    {
                        "chat_history": memories["social_memory"].load_memory_variables(
                            {}
                        )["chat_history"],
                        "input": f"Summarize social media sentiment for: {query}",
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
            with st.spinner("� Generating insights..."):
                insights_prompt = generate_insights_prompt(query, query_type)
                result = execute_agent_safely(
                    executors["insights"],
                    {
                        "chat_history": memories[
                            "insights_memory"
                        ].load_memory_variables({})["chat_history"],
                        "input": insights_prompt,
                    },
                )

                if result["success"]:
                    responses.append(f"## � Market Insights\n{result['output']}")
                    memories["insights_memory"].save_context(
                        {"input": insights_prompt}, {"output": result["output"]}
                    )
                else:
                    errors.append(f"❌ Insights Generation failed: {result['error']}")

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
                        "input": query,
                    },
                )

                if result["success"]:
                    final_response = f"## 🤖 General Analysis\n{result['output']}"
                else:
                    final_response = "❌ I apologize, but I'm unable to process your request at the moment. Please try again later."

        # Append errors if any
        if errors:
            st.session_state.error_count += len(errors)
            final_response += f"\n\n## ⚠️ Warnings\n" + "\n".join(errors)

        return final_response

    except Exception as e:
        logger.error(f"Multi-agent query failed: {e}")
        st.session_state.error_count += 1
        return f"❌ An unexpected error occurred: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."


def approve_hitl_response(user_query: str):
    """Process HITL approval with logging."""
    try:
        final_response = st.session_state.hitl_edit_box
        original_response = st.session_state.pending_hitl_response

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        diff = diff_text(original_response, final_response)

        st.session_state.hitl_log.append(
            {
                "query": user_query,
                "original_response": original_response,
                "edited_response": final_response,
                "timestamp": timestamp,
                "status": "approved",
                "diff": diff,
            }
        )

        # Add to conversation history
        st.session_state.conversation_history.extend(
            [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": final_response},
            ]
        )

        # Persist to storage
        append_history(
            [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": final_response},
            ]
        )

        st.session_state.latest_response = final_response
        st.session_state.pending_hitl_response = None

        # Show success message
        st.success("✅ Response approved and added to conversation!")

    except Exception as e:
        st.error(f"Error approving response: {e}")


def reject_hitl_response(user_query: str):
    """Discard the pending response and log the action."""
    try:
        st.session_state.hitl_log.append(
            {
                "query": user_query,
                "original_response": st.session_state.pending_hitl_response,
                "edited_response": None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "rejected",
                "diff": "",
            }
        )
        st.session_state.pending_hitl_response = None
        st.warning("❌ Response rejected and discarded.")
    except Exception as e:
        st.error(f"Error rejecting response: {e}")


def display_sidebar():
    """Display enhanced sidebar with controls and statistics."""
    with st.sidebar:
        st.markdown("## 🎛️ Controls")

        # HITL toggle
        st.session_state.hitl_mode = st.checkbox(
            "🧠 Human-in-the-Loop Review",
            value=st.session_state.hitl_mode,
            help="Review and edit AI responses before they're added to the conversation",
        )

        # Conversation management
        st.markdown("---")
        st.markdown("## 💬 Conversation")

        if st.button("🗑️ Clear History", help="Clear all conversation history"):
            st.session_state.conversation_history = []
            st.session_state.latest_response = ""
            st.session_state.last_user_query = ""
            clear_history()
            st.success("Conversation history cleared!")

        # HITL log viewer
        if st.session_state.hitl_log and st.checkbox("🔍 View HITL Edit Log"):
            st.markdown("### Recent HITL Actions")
            for i, entry in enumerate(reversed(st.session_state.hitl_log[-5:]), 1):
                label = "Edit" if entry.get("status") == "approved" else "Reject"
                with st.expander(f"{label} {i}: {entry['query'][:30]}..."):
                    st.markdown(f"**Timestamp:** {entry.get('timestamp','')}  ")
                    if entry.get("status") == "rejected":
                        st.markdown("Response was rejected.")
                        st.text_area(
                            "Original",
                            entry.get("original_response", ""),
                            height=100,
                            key=f"orig_{i}",
                            disabled=True,
                        )
                    else:
                        st.markdown("**Original:**")
                        st.text_area(
                            "",
                            entry.get("original_response", ""),
                            height=100,
                            key=f"orig_{i}",
                            disabled=True,
                        )
                        st.markdown("**Final:**")
                        st.text_area(
                            "",
                            entry.get("edited_response", ""),
                            height=100,
                            key=f"edit_{i}",
                            disabled=True,
                        )
                        if entry.get("diff"):
                            st.markdown("**Changes:**")
                            st.code(entry["diff"], language="diff")


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

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
        '<h1 class="main-header">📈 Stock Insight Assistant</h1>',
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
            "What would you like to know about the markets today?",
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

                    # Editable response
                    edited_response = st.text_area(
                        "Edit the response if needed:",
                        height=400,
                        key="hitl_edit_box",
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

                    # Add to conversation history
                    st.session_state.conversation_history.extend(
                        [
                            {"role": "user", "content": user_query},
                            {"role": "assistant", "content": generated_response},
                        ]
                    )
                    st.session_state.latest_response = generated_response

        elif st.session_state.hitl_mode and st.session_state.pending_hitl_response:
            # Show pending response (e.g., after regeneration)
            with st.chat_message("assistant"):
                st.markdown("### 💡 Review Assistant's Draft Response")
                st.info(
                    "Please review and edit the response below if needed, then click 'Approve' to add it to the conversation."
                )

                edited_response = st.text_area(
                    "Edit the response if needed:",
                    height=400,
                    key="hitl_edit_box",
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
