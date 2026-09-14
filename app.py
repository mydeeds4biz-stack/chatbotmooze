import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


st.set_page_config(page_title="Gemini Chatbot")
st.title("Gemini Chatbot")
st.caption("A simple conversational chatbot powered by Gemini.")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY is not configured. Add it in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-3.6-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Chat controls")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask Gemini something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = model.generate_content(prompt)
            answer = response.text
            st.markdown(answer)
        except ResourceExhausted:
            answer = (
                "Gemini's API quota has been reached. Please wait for the quota "
                "to reset or use an API key with available billing/quota."
            )
            st.warning(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
