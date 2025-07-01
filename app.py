import streamlit as st
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

from agents import *
from utils import classify_query, generate_insights_prompt

# Initialize LangChain's ChatGroq Model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# initialize memories for each agents
if "stock_memory" not in st.session_state:
    st.session_state.stock_memory = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history"
    )
if "sentiment_memory" not in st.session_state:
    st.session_state.sentiment_memory = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history"
    )
if "insights_memory" not in st.session_state:
    st.session_state.insights_memory = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history"
    )
# if "general_memory" not in st.session_state:
#     st.session_state.general_memory = ConversationBufferMemory(
#         return_messages=True, memory_key="chat_history"
#     )
# initialize memories for each agents
if "latest_response" not in st.session_state:
    st.session_state.latest_response = ""
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""
# initialize for hitl
if "hitl_mode" not in st.session_state:
    st.session_state.hitl_mode = True
if "pending_hitl_response" not in st.session_state:
    st.session_state.pending_hitl_response = None
if "hitl_log" not in st.session_state:
    st.session_state.hitl_log = []

# Initialize agents (replace `llm` with your actual LLM instance)
stock_data_agent = create_stock_data_agent(llm)
sentiment_agent = create_sentiment_agent(llm)
insights_agent = create_insights_agent(llm)
general_purpose_agent = create_general_purpose_agent(llm)
coordinator_agent = create_coordinator_agent(llm)

# Agent executors
stock_data_executor = AgentExecutor(
    agent=stock_data_agent,
    tools=[get_stock_data, get_stock_analysis],
    memory=st.session_state.stock_memory,
    verbose=True,
)
sentiment_executor = AgentExecutor(
    agent=sentiment_agent,
    tools=[get_market_sentiment_news, get_news_from_newsapi],
    verbose=True,
    memory=st.session_state.sentiment_memory,
)
insights_executor = AgentExecutor(
    agent=insights_agent,
    tools=[tavily_search, process_search_tool],
    verbose=True,
    memory=st.session_state.insights_memory,
)
# general_purpose_executor = AgentExecutor(
#     agent=general_purpose_agent,
#     tools=[tavily_search],
#     verbose=True,
#     memory=st.session_state.general_memory,
# )
# coordinator_executor = AgentExecutor(agent=coordinator_agent, tools=[], verbose=True)


def multi_agent_query(query):
    # Add the user query to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": query})

    responses = []
    errors = []
    query_type = classify_query(query)
    print(query_type)

    context = ""
    if len(st.session_state.conversation_history) > 1:
        previous_exchanges = st.session_state.conversation_history[
            -5:-1
        ]  # Get up to 4 previous exchanges
        for exchange in previous_exchanges:
            if exchange.get("role") == "assistant":
                context += f"Previous response: {exchange.get('content')}\n"

    # Fetch stock data if it's a stock-related query
    if query_type in ["stock", "both"]:
        try:
            stock_data_response = stock_data_executor.invoke(
                {
                    "chat_history": st.session_state.stock_memory.load_memory_variables(
                        {}
                    )["chat_history"],
                    "input": f"Retrieve the latest stock data and market trends for {query}. Provide key statistics, including open, high, low, close, and volume.",
                }
            )
            responses.append(
                f"**Stock Data Analysis**:\n{stock_data_response['output']}"
            )
            st.session_state.stock_memory.save_context(
                {"input": query}, {"output": stock_data_response["output"]}
            )
        except Exception as e:
            errors.append(f"❌ Stock Data Agent failed: {str(e)}")

    # Fetch sentiment analysis if it's related to financial sentiment
    if query_type in ["sentiment", "both"]:
        try:
            sentiment_response = sentiment_executor.invoke(
                {
                    "chat_history": st.session_state.sentiment_memory.load_memory_variables(
                        {}
                    )[
                        "chat_history"
                    ],
                    "input": f"Analyze the market sentiment for {query}. Summarize the tone of recent news articles, social media discussions, and investor opinions.",
                }
            )
            responses.append(f"**Sentiment Analysis**:\n{sentiment_response['output']}")
            # Save to memory
            st.session_state.sentiment_memory.save_context(
                {"input": query}, {"output": sentiment_response["output"]}
            )
        except Exception as e:
            errors.append(f"❌ Sentiment Agent failed: {str(e)}")

    # Step 3: Generate insights (generic input)e
    if query_type in ["stock", "sentiment", "both", "general"]:
        try:
            insights_prompt = generate_insights_prompt(query, query_type)
            insights_response = insights_executor.invoke(
                {
                    "chat_history": st.session_state.insights_memory.load_memory_variables(
                        {}
                    )[
                        "chat_history"
                    ],
                    "input": insights_prompt,
                }
            )
            responses.append(f"**Insights**:\n{insights_response['output']}")
            # Save to memory
            st.session_state.insights_memory.save_context(
                {"input": insights_prompt}, {"output": insights_response["output"]}
            )
        except Exception as e:
            errors.append(f"❌ Insights Agent failed: {str(e)}")

    # Combine responses
    final_response = "\n\n".join(responses) if responses else "No data available."

    # Append errors if any agents failed
    if errors:
        final_response += "\n\n**Errors**:\n" + "\n".join(errors)
    return final_response


def display_chat_history():
    if "conversation_history" in st.session_state:
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])


# Initialize session state for storing conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit UI
st.set_page_config(page_title="Stock Assistant", page_icon="📈")
st.title("📈 Stock Insight Assistant")
st.markdown(
    "Ask me about a stock’s fundamentals, news sentiment, or technicals. I’m multi-agent powered! 🤖"
)

# -- Sidebar toggle --
with st.sidebar:
    st.session_state.hitl_mode = st.checkbox(
        "🧠 Human-in-the-Loop Review", value=st.session_state.hitl_mode
    )


# -- HITL approval function with logging --
def approve_hitl_response(user_query):
    final_response = st.session_state.hitl_edit_box
    original_response = st.session_state.pending_hitl_response

    # Log the edit
    if final_response != original_response:
        st.session_state.hitl_log.append(
            {
                "query": user_query,
                "original_response": original_response,
                "edited_response": final_response,
            }
        )

    st.session_state.conversation_history.append(
        {"role": "user", "content": user_query}
    )
    st.session_state.conversation_history.append(
        {"role": "assistant", "content": final_response}
    )
    st.session_state.latest_response = final_response
    st.session_state.pending_hitl_response = None


# -- Main input and output display logic with HITL injection --
user_query = st.chat_input(
    "Which stock do you want to know about today?", key="stock_question"
)

if user_query and user_query.strip():
    st.chat_message("user").write(user_query)
    st.session_state.last_user_query = user_query

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            generated_response = multi_agent_query(user_query)

        if st.session_state.hitl_mode:
            st.session_state.pending_hitl_response = generated_response
            st.markdown("#### 💡 Review Assistant's Draft Response:")
            st.text_area(
                "Edit the response if needed",
                value=generated_response,
                key="hitl_edit_box",
                height=300,
            )
            st.button(
                "✅ Approve & Send to Chat",
                on_click=lambda: approve_hitl_response(user_query),
            )
        else:
            st.session_state.conversation_history.append(
                {"role": "user", "content": user_query}
            )
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": generated_response}
            )
            st.session_state.latest_response = generated_response
            st.markdown(generated_response)

# -- Show conversation history after response --
if "conversation_history" in st.session_state:
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

# -- debug: view HITL log
if st.sidebar.checkbox("🔍 View HITL Edit Log"):
    for entry in st.session_state.hitl_log:
        st.sidebar.markdown(f"**Query:** {entry['query']}")
        st.sidebar.markdown(f"**Original:**\n{entry['original_response']}")
        st.sidebar.markdown(f"**Edited:**\n{entry['edited_response']}")
        st.sidebar.markdown("---")
