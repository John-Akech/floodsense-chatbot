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
    """Apply custom CSS for modern professional interface."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: white;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1f2937;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .main .block-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1.5rem;
        background: white;
    }
    
    .message-container {
        margin: 2rem 0;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .user-message {
        background: #f8fafc;
        color: #1f2937;
        margin-left: auto;
        margin-right: 0;
        border-left: 4px solid #3b82f6;
        text-align: left;
    }
    
    .assistant-message {
        background: #f9fafb;
        color: #1f2937;
        margin-left: 0;
        margin-right: auto;
        border-left: 4px solid #10b981;
        text-align: left;
    }
    
    .message-text {
        line-height: 1.6;
        font-size: 15px;
        color: #1f2937;
    }
    
    .welcome-message {
        text-align: center;
        padding: 5rem 3rem;
        color: #1f2937;
        margin: 3rem 0;
    }
    
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 2rem;
        color: #1f2937;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .welcome-subtitle {
        font-size: 1.4rem;
        font-weight: 400;
        margin-bottom: 3rem;
        color: #6b7280;
        line-height: 1.5;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .sidebar-content {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .sidebar-title {
        color: #1f2937;
        font-weight: 800;
        font-size: 1.4rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-subtitle {
        color: #6b7280;
        font-weight: 400;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .stButton > button {
        background: white !important;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-bottom: 0.75rem !important;
        font-size: 14px !important;
        text-align: left !important;
    }
    
    .stButton > button:hover {
        background: #f8fafc !important;
        border-color: #3b82f6 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    .stMetric {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        padding: 0.75rem !important;
    }
    
    .stMetric > div {
        color: #1f2937 !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox > div > div {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        color: #1f2937 !important;
    }
    
    /* All text elements */
    .stMarkdown, .stText, p, div {
        color: #1f2937 !important;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: #f8fafc !important;
        padding: 1.5rem !important;
    }
    
    .stSidebar .stMarkdown {
        color: #1f2937 !important;
        margin-bottom: 1.5rem !important;
    }
    
    .stSidebar hr {
        margin: 2rem 0 !important;
        border-color: #e5e7eb !important;
    }
    
    /* Chat messages container */
    .chat-messages {
        padding: 2rem 0;
        margin-bottom: 3rem;
    }
    
    /* Professional footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6b7280;
        font-size: 12px;
        border-top: 1px solid #e5e7eb;
        margin-top: 3rem;
    }
    
    /* Loading spinner customization */
    .stSpinner {
        text-align: center;
        color: #3b82f6 !important;
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
        page_title="FloodSense - Climate & Flood Risk Assistant",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "FloodSense - AI Assistant for Flood Risk & Climate Information in South Sudan"
        }
    )
    
    # Initialize session state first
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
            st.rerun()
        
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
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"delete_{i}"):
                        st.session_state.chat_history.pop(i)
                        try:
                            with open("chat_history/history.json", "w") as f:
                                json.dump(st.session_state.chat_history, f, indent=2)
                        except:
                            pass
                        st.rerun()
            
            if st.button("Clear All History"):
                clear_all_history()
                st.rerun()
        else:
            st.write("No chat history")
        
        # Clear current chat button
        if st.button("Clear Current Chat"):
            st.session_state.messages = []
            st.rerun()
        
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
                # Ensure messages list exists
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({"role": "user", "content": selected_question})
                response = generate_response(selected_question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Statistics**")
        
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", total_messages)
        with col2:
            st.metric("Questions", user_messages)
        
        st.markdown("---")
        st.markdown("**Information Center**")
        
        if st.button("Regional Assessments", key="sidebar_regional"):
            question = "Regional Assessments"
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": question})
            response = generate_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Safety Guidelines", key="sidebar_safety"):
            question = "Safety Guidelines"
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": question})
            response = generate_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("Climate Information", key="sidebar_climate"):
            question = "Climate Information"
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": question})
            response = generate_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Main chat area
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-message">
            <div class="welcome-title">FloodSense</div>
            <div class="welcome-subtitle">AI Assistant for Flood Risk & Climate Information in South Sudan</div>
            <div class="welcome-examples">
                <p style="color: #6b7280; font-size: 14px; margin-top: 2rem;">Ask about flood risks, climate impacts, safety guidelines, or regional assessments for any location in South Sudan</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="message-container user-message">
                    <div class="message-text">{message["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="message-container assistant-message">
                    <div class="message-text">{message["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input form with Enter key support
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            prompt = st.text_input(
                "Input",
                placeholder="Ask FloodSense anything about South Sudan floods...",
                label_visibility="collapsed",
                key="user_input"
            )
        with col2:
            submitted = st.form_submit_button("Send")
    
    # Input CSS styling
    st.markdown("""
    <style>
    .stTextInput input {
        background: white !important;
        border: 1px solid #d1d5db !important;
        border-radius: 16px !important;
        padding: 16px 24px !important;
        font-size: 16px !important;
        font-weight: 400 !important;
        color: #1f2937 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }
    
    .stTextInput input:focus {
        background: white !important;
        border: 2px solid #3b82f6 !important;
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1), 0 4px 16px rgba(0,0,0,0.08) !important;
        transform: translateY(-1px) !important;
    }
    
    .stTextInput input::placeholder {
        color: #9ca3af !important;
        font-weight: 400 !important;
    }
    
    .stButton[data-testid="baseButton-secondary"] > button {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 16px 24px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stButton[data-testid="baseButton-secondary"] > button:hover {
        background: #2563eb !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stForm {
        background: white !important;
        border-top: 1px solid #e5e7eb !important;
        padding: 2rem 0 !important;
        margin-top: 3rem !important;
        position: sticky !important;
        bottom: 0 !important;
        box-shadow: 0 -4px 12px rgba(0,0,0,0.05) !important;
    }
    
    .stForm > div {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 0 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Process input
    if submitted and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🤖 FloodSense is analyzing your question..."):
            try:
                response = generate_response(prompt)
                if not response or response.strip() == "":
                    response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question about flood risks or climate information in South Sudan."
            except Exception as e:
                response = f"I encountered an error while processing your question. Please try again or contact support if the issue persists."
                st.error(f"Error: {str(e)}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>FloodSense</strong> - Climate & Flood Risk Information System for South Sudan</p>
        <p>Providing comprehensive flood risk assessments, climate change information, and emergency preparedness guidance</p>
        <p>© 2025 FloodSense Chatbot | Built with advanced AI for climate resilience</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()