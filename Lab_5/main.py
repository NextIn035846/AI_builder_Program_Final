import streamlit as st
from dotenv import load_dotenv
from typing import Set, List, Dict, Any
from core import run_llm2
from io import BytesIO
import requests
from PIL import Image

# Load environment variables (should be at the very top)
load_dotenv()

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(
    page_title="LangChain RAG Bot",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---

def create_sources_string(source_urls: Set[str]) -> str:
    """Formats a set of source URLs into a numbered string."""
    if not source_urls:
        return ""
    sources_list = sorted(list(source_urls))
    return "sources:\n" + "\n".join(f"{i+1}. {source}" for i, source in enumerate(sources_list))

def get_profile_picture(email):
    """Fetches a profile picture from Gravatar."""
    try:
        gravatar_url = f"https://www.gravatar.com/avatar/{hash(email)}?d=identicon&s=150"
        response = requests.get(gravatar_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return Image.open(BytesIO(response.content))
    except requests.RequestException:
        # Return a default image or placeholder if Gravatar fails
        return Image.new('RGB', (150, 150), color = 'grey')


# --- Custom Styling ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #F9F9F9; /* Very light grey background */
        color: #333333; /* Dark grey text for softer contrast */
    }
    
    /* Text input box */
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #DDDDDD; /* Light border */
    }
    
    /* Button */
    .stButton > button {
        background-color: #4CAF50; /* The green you had works well */
        color: #FFFFFF;
        border: none;
    }
    
    /* Sidebar */
    .stSidebar {
        background-color: #F1F1F1; /* Slightly darker grey for the sidebar */
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("User Profile")
    user_name = "Thomas Patole"
    user_email = "thomaspatole19@gmail.com"
    profile_pic = get_profile_picture(user_email)
    st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")
    st.markdown("---")
    
    # IMPORTANT: Button to clear the chat history
    if st.button("Clear Chat History"):
        st.session_state.clear() # Clears all session state variables
        st.rerun() # Reruns the app to reflect the cleared state immediately


# --- Session State Initialization ---
# This ensures that the keys exist in the session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# --- Main Page ---
st.header("LangChain🦜🔗 Udemy Course- Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your question here...")

if prompt:
    with st.spinner("Generating response..."):
        # Call the language model
        response_dict = run_llm2(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        # Safely get the answer and context
        answer = response_dict.get("answer", "Sorry, I couldn't generate a response.")
        context = response_dict.get("context", [])
        
        # Ensure 'answer' is a string before proceeding
        if not isinstance(answer, str):
            answer = str(answer) # Convert to string just in case

        sources = set(doc.metadata.get("source", "Unknown") for doc in context)
        formatted_response = f"{answer}\n\n{create_sources_string(sources)}"

        # Update session state history
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", answer)) # Append the clean answer string

# --- Display Chat History ---
# This loop is now robust and will not crash the app
if st.session_state["chat_answers_history"]:
    for user_query, generated_response in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answers_history"],
    ):
        st.chat_message("user").write(user_query)
        
        # Defensive check: Ensure the response is a string before writing
        if isinstance(generated_response, str):
            st.chat_message("assistant").write(generated_response)
        else:
            # If a non-string object is found, display an error instead of crashing
            st.chat_message("assistant").error(
                f"Invalid response format found in history: {type(generated_response)}"
            )

# --- Footer ---
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")