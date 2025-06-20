"""
FloodSense: South Sudan - Main Application
A domain-specific chatbot for flood risk information in South Sudan.
"""
import streamlit as st
import json
import os
import datetime
from infer import generate_response

def apply_custom_css():
    """Apply custom CSS for ChatGPT-like interface."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background-color: #343541;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .main .block-container {
        padding: 0;
        max-width: none;
    }
    
    .chat-messages {
        height: calc(100vh - 140px);
        overflow-y: auto;
        padding: 0;
    }
    
    .message-container {
        width: 100%;
        padding: 20px 0;
        border-bottom: 1px solid #444654;
    }
    
    .user-message {
        background-color: #343541;
    }
    
    .assistant-message {
        background-color: #444654;
    }
    
    .message-content {
        max-width: 768px;
        margin: 0 auto;
        padding: 0 20px;
        display: flex;
        gap: 16px;
    }
    
    .message-avatar {
        width: 30px;
        height: 30px;
        border-radius: 2px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background-color: #5436da;
        color: white;
    }
    
    .assistant-avatar {
        background-color: #19c37d;
        color: white;
    }
    
    .message-text {
        color: #ececf1;
        line-height: 1.6;
        font-size: 16px;
        flex: 1;
        padding-top: 4px;
    }
    
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #343541;
        padding: 20px;
        border-top: 1px solid #444654;
    }
    
    .input-wrapper {
        max-width: 768px;
        margin: 0 auto;
        position: relative;
    }
    
    .stTextInput > div > div > input {
        background: #40414f !important;
        border: 1px solid #565869 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        color: #ececf1 !important;
        font-size: 16px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #19c37d !important;
        outline: none !important;
        box-shadow: 0 0 0 1px #19c37d !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #8e8ea0 !important;
    }
    
    .stButton > button {
        background: #19c37d !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
    }
    
    .stButton > button:hover {
        background: #0d8f6f !important;
    }
    
    .css-1d391kg {
        background-color: #202123 !important;
    }
    
    .sidebar-content {
        color: #ececf1;
        padding: 16px;
    }
    
    .sidebar-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #ececf1;
    }
    
    .sidebar-subtitle {
        font-size: 14px;
        color: #8e8ea0;
        margin-bottom: 20px;
    }
    
    .stSelectbox > div > div {
        background: #40414f !important;
        border: 1px solid #565869 !important;
        color: #ececf1 !important;
    }
    
    .stMetric {
        background: #40414f !important;
        padding: 8px !important;
        border-radius: 6px !important;
        border: 1px solid #565869 !important;
    }
    
    .stMetric > div {
        color: #ececf1 !important;
    }
    
    .welcome-message {
        max-width: 768px;
        margin: 40px auto;
        padding: 0 20px;
        text-align: center;
        color: #ececf1;
    }
    
    .welcome-title {
        font-size: 32px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    .welcome-subtitle {
        font-size: 18px;
        color: #8e8ea0;
        margin-bottom: 32px;
    }
    
    .welcome-examples {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 16px;
        margin-top: 32px;
    }
    
    .example-card {
        background: #444654;
        border: 1px solid #565869;
        border-radius: 8px;
        padding: 16px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .example-card:hover {
        background: #40414f;
    }
    
    .example-title {
        font-weight: 600;
        margin-bottom: 8px;
        color: #ececf1;
    }
    
    .example-text {
        font-size: 14px;
        color: #8e8ea0;
    }
    
    .chat-messages::-webkit-scrollbar {
        width: 4px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: #343541;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #565869;
        border-radius: 2px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #8e8ea0;
    }
    
    .stForm {
        background: transparent !important;
        border: none !important;
    }
    
    .stForm > div {
        background: transparent !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

def save_chat_history():
    """Save current chat to history."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if len(st.session_state.messages) > 0:
        chat_session = {
            "id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.messages.copy(),
            "title": st.session_state.messages[0]["content"][:50] + "..." if len(st.session_state.messages[0]["content"]) > 50 else st.session_state.messages[0]["content"]
        }
        st.session_state.chat_history.insert(0, chat_session)
        
        # Save to file
        try:
            os.makedirs("chat_history", exist_ok=True)
            with open("chat_history/history.json", "w") as f:
                json.dump(st.session_state.chat_history, f, indent=2)
        except Exception as e:
            st.error(f"Error saving chat history: {e}")

def load_chat_history():
    """Load chat history from file."""
    try:
        if os.path.exists("chat_history/history.json"):
            with open("chat_history/history.json", "r") as f:
                st.session_state.chat_history = json.load(f)
    except Exception as e:
        st.session_state.chat_history = []

def clear_all_history():
    """Clear all chat history."""
    st.session_state.chat_history = []
    try:
        if os.path.exists("chat_history/history.json"):
            os.remove("chat_history/history.json")
    except Exception as e:
        st.error(f"Error clearing history: {e}")

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="FloodSense: South Sudan",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Load chat history
    load_chat_history()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <div class="sidebar-title">FloodSense</div>
            <div class="sidebar-subtitle">South Sudan Flood Information</div>
        </div>
        """, unsafe_allow_html=True)
        
        # New conversation button
        if st.button("+ New Conversation"):
            save_chat_history()
            st.session_state.messages = []
            st.experimental_rerun()
        
        # Chat history section
        st.markdown("---")
        st.markdown("**Chat History**")
        
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history[:10]):  # Show last 10 chats
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"{chat['title']}", key=f"load_{i}"):
                        save_chat_history()
                        st.session_state.messages = chat["messages"].copy()
                        st.experimental_rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{i}"):
                        st.session_state.chat_history.pop(i)
                        try:
                            with open("chat_history/history.json", "w") as f:
                                json.dump(st.session_state.chat_history, f, indent=2)
                        except:
                            pass
                        st.experimental_rerun()
            
            if st.button("Clear All History"):
                clear_all_history()
                st.experimental_rerun()
        else:
            st.write("No chat history")
        
        st.markdown("---")
        st.markdown("**Quick Questions**")
        
        quick_questions = [
            "Select a question...",
            "What are the main causes of flooding in South Sudan?",
            "How does the government coordinate flood response?",
            "What is South Sudan's national flood risk mapping?",
            "How does climate change impact flooding nationally?",
            "What disaster financing exists for floods?",
            "What is the flood risk in Jonglei State?",
            "What is the flood risk in Unity State?",
            "What is the flood risk in Upper Nile State?",
            "How do rural and urban flood risks compare?",
            "What are the regional early warning gaps?",
            "What are South Sudan's rainfall patterns?",
            "How do CHIRPS satellite observations help?",
            "What causes White Nile river overflow?",
            "How does climate change intensify floods?",
            "How should households prepare for floods?",
            "What are emergency evacuation procedures?",
            "What should be in a flood survival kit?",
            "How to ensure safe water during floods?",
            "How to protect vulnerable populations?",
            "How often do roads collapse during floods?",
            "How do floods affect power and communication?",
            "What happens to schools and clinics?",
            "How to access remote areas after floods?",
            "How do floods destroy crops like sorghum?",
            "What happens to livestock during floods?",
            "How do floods affect soil fertility?",
            "How do floods disrupt planting seasons?",
            "What waterborne diseases occur after floods?",
            "How do floods affect healthcare access?",
            "How do floods cause malnutrition?",
            "What are the mental health effects?",
            "How do floods cause school dropouts?",
            "How do floods affect employment?",
            "What happens to housing after floods?",
            "How do floods create poverty cycles?",
            "How do UN and NGOs coordinate relief?",
            "Where are emergency shelters located?",
            "What are the logistics challenges?",
            "What recovery and resilience programs exist?"
        ]
        
        selected_question = st.selectbox(
            "Choose a question:",
            quick_questions,
            key="quick_question_selector"
        )
        
        if st.button("Ask Question", disabled=(selected_question == "Select a question...")):
            if selected_question != "Select a question...":
                st.session_state.messages.append({"role": "user", "content": selected_question})
                response = generate_response(selected_question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("**Statistics**")
        
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", total_messages)
        with col2:
            st.metric("Questions", user_messages)
    
    # Main chat area
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-message">
            <div class="welcome-title">FloodSense</div>
            <div class="welcome-subtitle">AI Assistant for Flood Risk Information in South Sudan</div>
            <div class="welcome-examples">
                <div class="example-card">
                    <div class="example-title">Regional Assessments</div>
                    <div class="example-text">Get flood risk information for specific regions like Bentiu, Bor, Malakal, and Juba</div>
                </div>
                <div class="example-card">
                    <div class="example-title">Safety Guidelines</div>
                    <div class="example-text">Learn emergency preparation and safety measures during flooding</div>
                </div>
                <div class="example-card">
                    <div class="example-title">Climate Information</div>
                    <div class="example-text">Understand seasonal patterns and climate change impacts</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="message-container user-message">
                    <div class="message-content">
                        <div class="message-avatar user-avatar">U</div>
                        <div class="message-text">{message["content"]}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="message-container assistant-message">
                    <div class="message-content">
                        <div class="message-avatar assistant-avatar">F</div>
                        <div class="message-text">{message["content"]}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Fixed input at bottom
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    
    # Working Enter key form
    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_input(
            "Message FloodSense...",
            placeholder="Type your message and press Enter...",
            label_visibility="collapsed",
            key="user_input"
        )
        submit_button = st.form_submit_button("Send", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process input - Enter key works with st.form automatically
    if submit_button and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()

if __name__ == "__main__":
    main()