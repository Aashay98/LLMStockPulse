from datetime import datetime

import streamlit as st

from storage import append_history
from utils import diff_text


def approve_hitl_response(user_query: str) -> None:
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

        st.session_state.conversation_history.extend(
            [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": final_response},
            ]
        )

        append_history(
            [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": final_response},
            ],
            st.session_state.get("user_id", "default"),
            st.session_state.get("current_conversation"),
        )

        st.session_state.latest_response = final_response
        st.session_state.pending_hitl_response = None

        st.success("✅ Response approved and added to conversation!")

    except Exception as e:
        st.error(f"Error approving response: {e}")


def reject_hitl_response(user_query: str) -> None:
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
