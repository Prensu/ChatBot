import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/generate"

st.title("ChatBot (FastAPI + Streamlit)")

user_input = st.text_area("You:", "")

if st.button("Send"):
    if user_input.strip():
        payload = {"prompt": user_input}
        response = requests.post(BACKEND_URL, json=payload)
        
        if response.status_code == 200:
            bot_reply = response.json()["response"]
            st.write("**Bot:**", bot_reply)
        else:
            st.error("Error contacting backend.")
