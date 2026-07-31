"""
chatbot_widget.py — Streamlit floating / sidebar interactive chatbot widget for ArecaVision AI.

Renders an interactive AI Agronomist Chatbot widget ("ArecaBot") in the Streamlit application.
"""

import streamlit as st
from recommendation.chatbot_engine import get_bot_response, DEFAULT_RESPONSES


def init_chat_session():
    """Initializes chatbot session state variables if not already present."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "sender": "bot",
                "text": "🌴 **Namaskara! I am ArecaBot**, your AI Agronomist Assistant.\n\nAsk me anything about **Areca nut diseases, leaf health, Bordeaux mixture preparation, organic remedies, or fertilizer schedules**!"
            }
        ]


def render_chatbot_widget(current_prediction: str = None):
    """
    Renders the ArecaBot Chatbot UI panel in Streamlit sidebar or page.
    """
    init_chat_session()

    st.markdown("""
    <style>
        .chat-card {
            background-color: #1A2420;
            border: 1px solid #2E7559;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .user-msg {
            background-color: #24352D;
            border-left: 4px solid #3C9370;
            color: #FFFFFF;
            padding: 10px 14px;
            border-radius: 8px;
            margin: 6px 0;
        }
        .bot-msg {
            background-color: #161D19;
            border-left: 4px solid #2E7559;
            color: #E0E0E0;
            padding: 10px 14px;
            border-radius: 8px;
            margin: 6px 0;
        }
        .chip-btn button {
            background-color: #1C2822 !important;
            border: 1px solid #2E7559 !important;
            color: #A3D9C9 !important;
            font-size: 12px !important;
            padding: 4px 10px !important;
            border-radius: 15px !important;
        }
        .chip-btn button:hover {
            background-color: #2E7559 !important;
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.expander("💬 **Ask ArecaBot (AI Agronomist Assistant)**", expanded=False):
        st.caption("Ask questions about Areca palm diseases, organic remedies, Bordeaux mixture, or fertilizers.")

        if current_prediction and current_prediction != "Unknown":
            st.info(f"💡 **Active Context**: Current scan diagnosed as **{current_prediction}**.")

        # Quick action chips
        st.markdown("**Quick Questions:**")
        chip_col1, chip_col2 = st.columns(2)

        selected_chip = None
        if chip_col1.button("🧪 1% Bordeaux Recipe", key="chip_bordeaux"):
            selected_chip = "How to prepare 1% Bordeaux mixture step by step?"
        if chip_col2.button("🌧️ Koleroga Treatment", key="chip_koleroga"):
            selected_chip = "How to treat Koleroga or Mahali rot during monsoon?"

        chip_col3, chip_col4 = st.columns(2)
        if chip_col3.button("🍃 Yellow Leaf Care", key="chip_yld"):
            selected_chip = "What is the organic treatment for Yellow Leaf Disease?"
        if chip_col4.button("🌱 NPK Fertilizer Dose", key="chip_npk"):
            selected_chip = "What is the recommended fertilizer schedule for Areca palms?"

        st.write("---")

        # Render conversation history
        chat_container = st.container()
        with chat_container:
            for idx, msg in enumerate(st.session_state.chat_history):
                if msg["sender"] == "user":
                    st.markdown(f"<div class='user-msg'>👨‍🌾 <b>You:</b> {msg['text']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot-msg'>🤖 <b>ArecaBot:</b><br>{msg['text']}</div>", unsafe_allow_html=True)

        # Input box for farmer query
        with st.form(key="arecabot_chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask ArecaBot a question:",
                value=selected_chip if selected_chip else "",
                placeholder="e.g. How to prepare 1% Bordeaux mixture?",
                key="chat_input_field"
            )
            submit_col, clear_col = st.columns([4, 1])
            submitted = submit_col.form_submit_button("Send ➔")
            cleared = clear_col.form_submit_button("Clear")

        if cleared:
            st.session_state.chat_history = [
                {
                    "sender": "bot",
                    "text": "🌴 **Chat history cleared.** Ask me anything about Areca nut and leaf health!"
                }
            ]
            st.rerun()

        prompt_to_process = selected_chip if selected_chip else (user_input.strip() if submitted else None)

        if prompt_to_process:
            # Append user message
            st.session_state.chat_history.append({"sender": "user", "text": prompt_to_process})

            # Generate response from chatbot engine
            response_data = get_bot_response(prompt_to_process, current_prediction=current_prediction)
            st.session_state.chat_history.append({"sender": "bot", "text": response_data["answer"]})

            st.rerun()
