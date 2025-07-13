import streamlit as st
from langchain.memory import ConversationBufferWindowMemory

import config
from database import create_user, reset_password, verify_user
from storage import clear_history, create_conversation, get_conversations, load_history


def login_screen() -> None:
    """Login form that checks credentials against the database."""
    """Login and registration form."""
    st.markdown("## 🔐 Login")
    mode = st.radio("Mode", ["Login", "Register", "Reset Password"], horizontal=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if mode == "Register":
        password_confirm = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            if password != password_confirm:
                st.error("Passwords do not match")
            else:
                if create_user(username, password):
                    st.success("User created! Please log in.")
                else:
                    st.error("Username already exists")
        st.stop()

    elif mode == "Reset Password":
        new_pass = st.text_input("New Password", type="password")
        confirm = st.text_input("Confirm New Password", type="password")
        if st.button("Reset"):
            if new_pass != confirm:
                st.error("Passwords do not match")
            else:
                if reset_password(username, password, new_pass):
                    st.success("Password updated. Please log in.")
                else:
                    st.error("Invalid username or password")
        st.stop()

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state.authenticated = True
            st.session_state.user_id = username
            convs = get_conversations(username)
            if not convs:
                conv_id = create_conversation(username, "Chat 1")
                convs = get_conversations(username)
                st.session_state.current_conversation = conv_id
            else:
                st.session_state.current_conversation = convs[0]["id"]
            st.session_state.conversations = convs
            st.session_state.conversation_history = load_history(
                username,
                st.session_state.current_conversation,
            )
            st.success("Logged in!")
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()


def display_sidebar() -> None:
    """Display sidebar with controls and statistics."""
    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.get('user_id','guest')}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = ""
            st.rerun()
        st.markdown("## 🎛️ Controls")

        st.session_state.hitl_mode = st.checkbox(
            "🧠 Human-in-the-Loop Review",
            value=st.session_state.hitl_mode,
            help="Review and edit AI responses before they're added to the conversation",
        )

        st.markdown("---")
        st.markdown("## 💬 Conversation")
        conv_opts = {
            c["title"]: c["id"] for c in st.session_state.get("conversations", [])
        }
        if conv_opts:
            titles = list(conv_opts.keys())
            current_id = st.session_state.get("current_conversation")
            try:
                index = list(conv_opts.values()).index(current_id)
            except ValueError:
                index = 0
            selected_title = st.selectbox("Select chat", titles, index=index)
            selected_id = conv_opts[selected_title]
            if selected_id != current_id:
                st.session_state.current_conversation = selected_id
                st.session_state.conversation_history = load_history(
                    st.session_state.user_id,
                    selected_id,
                )
                for mkey in [
                    "stock_memory",
                    "sentiment_memory",
                    "social_memory",
                    "insights_memory",
                    "general_memory",
                ]:
                    st.session_state[mkey] = ConversationBufferWindowMemory(
                        k=config.MEMORY_WINDOW_SIZE,
                        return_messages=True,
                        memory_key="chat_history",
                    )
                st.rerun()

        if st.button("➕ New Chat"):
            new_title = f"Chat {len(conv_opts) + 1}"
            new_id = create_conversation(st.session_state.user_id, new_title)
            st.session_state.conversations.append({"id": new_id, "title": new_title})
            st.session_state.current_conversation = new_id
            st.session_state.conversation_history = []
            for mkey in [
                "stock_memory",
                "sentiment_memory",
                "social_memory",
                "insights_memory",
                "general_memory",
            ]:
                st.session_state[mkey] = ConversationBufferWindowMemory(
                    k=config.MEMORY_WINDOW_SIZE,
                    return_messages=True,
                    memory_key="chat_history",
                )
            st.rerun()

        if st.button("🗑️ Clear History", help="Clear all conversation history"):
            st.session_state.conversation_history = []
            st.session_state.latest_response = ""
            st.session_state.last_user_query = ""
            clear_history(
                st.session_state.get("user_id", "default"),
                st.session_state.get("current_conversation"),
            )
            st.success("Conversation history cleared!")

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
                        st.markdown("**Original:")
                        st.text_area(
                            "",
                            entry.get("original_response", ""),
                            height=100,
                            key=f"orig_{i}",
                            disabled=True,
                        )
                        st.markdown("**Final:")
                        st.text_area(
                            "",
                            entry.get("edited_response", ""),
                            height=100,
                            key=f"edit_{i}",
                            disabled=True,
                        )
                        if entry.get("diff"):
                            st.markdown("**Changes:")
                            st.code(entry["diff"], language="diff")
