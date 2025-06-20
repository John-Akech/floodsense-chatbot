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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .main .block-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem 1rem;
        background: rgba(255,255,255,0.98);
        border-radius: 16px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    .message-container {
        margin: 1.5rem 0;
        padding: 1.5rem;
        border-radius: 12px;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        margin-left: 3rem;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.25);
        border-left: 4px solid #60a5fa;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        margin-right: 3rem;
        box-shadow: 0 8px 25px rgba(220, 38, 38, 0.25);
        border-left: 4px solid #f87171;
    }
    
    .message-text {
        line-height: 1.7;
        font-size: 16px;
        font-weight: 500;
    }
    
    .welcome-message {
        text-align: center;
        padding: 3rem;
        color: #1e40af;
    }
    
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1e40af;
    }
    
    .welcome-subtitle {
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 2rem;
        color: #dc2626;
    }
    
    .sidebar-content {
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    .sidebar-title {
        color: #1e40af;
        font-weight: 700;
        font-size: 1.4rem;
    }
    
    .sidebar-subtitle {
        color: #dc2626;
        font-weight: 500;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.3) !important;
    }
    
    .stMetric > div {
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.95) !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 8px !important;
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
        page_icon="F",
        layout="wide",
        initial_sidebar_state="expanded"
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
                    if st.button("Delete", key=f"delete_{i}"):
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
        
        # Clear current chat button
        if st.button("Clear Current Chat"):
            st.session_state.messages = []
            st.experimental_rerun()
        
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
        
        st.markdown("---")
        st.markdown("**Information Center**")
        
        if st.button("Regional Assessments", key="sidebar_regional"):
            question = "Regional Assessments"
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": question})
            response = generate_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()
        
        if st.button("Safety Guidelines", key="sidebar_safety"):
            question = "Safety Guidelines"
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": question})
            response = generate_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()
        
        if st.button("Climate Information", key="sidebar_climate"):
            question = "Climate Information"
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": question})
            response = generate_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.experimental_rerun()
    
    # Main chat area
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-message">
            <div class="welcome-title">FloodSense</div>
            <div class="welcome-subtitle">AI Assistant for Flood Risk Information in South Sudan</div>
            <div class="welcome-examples">
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
    
    # Premium input design
    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_input(
            "Input",
            placeholder="Ask FloodSense anything about South Sudan floods...",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Send")
        
    # Professional input CSS with Enter key support
    st.markdown("""
    <style>
    .stTextInput input {
        background: rgba(255,255,255,0.98) !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #1f2937 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        background: rgba(255,255,255,1) !important;
        border: 2px solid #3b82f6 !important;
        outline: none !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.15), 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    .stTextInput input::placeholder {
        color: #6b7280 !important;
        font-weight: 500 !important;
    }
    
    .stForm button[type="submit"] {
        position: absolute;
        left: -9999px;
        width: 1px;
        height: 1px;
    }
    
    .stForm {
        position: sticky;
        bottom: 0;
        background: rgba(255,255,255,0.98);
        padding: 20px 0;
        border-top: 2px solid #e5e7eb;
        margin-top: 30px;
        border-radius: 12px 12px 0 0;
    }
    </style>
    
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const form = document.querySelector('[data-testid="stForm"]');
        if (form) {
            const input = form.querySelector('input');
            if (input) {
                input.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const submitBtn = form.querySelector('button[type="submit"]');
                        if (submitBtn) {
                            submitBtn.click();
                        }
                    }
                });
            }
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Process input
    if submitted and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()

if __name__ == "__main__":
    main()