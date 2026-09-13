import streamlit as st
import google.generativeai as genai


st.set_page_config(page_title="Gemini Chatbot")

st.title("Gemini Chatbot")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY is not configured. Add it in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-3.6-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    response = model.generate_content(prompt)
    answer = response.text

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
