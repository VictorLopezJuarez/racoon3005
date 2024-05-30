import os
import json
import streamlit as st

# Retrieve the Google Cloud credentials from Streamlit secrets
google_credentials = st.secrets["google_application_credentials"]

# Define the path to save the credentials file
credentials_path = "/tmp/google-credentials.json"

# Write the credentials to the file
with open(credentials_path, "w") as file:
    json.dump(google_credentials, file)

# Set the environment variable to the path of the credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Import the rest of your libraries and code here
from langchain_helper import create_qa_chain

def get_and_display_response(question):
    chain = create_qa_chain()
    response = chain(question)
    st.write(response)

st.title('AI Chatbot')
question = st.text_input('Ask a question:')
if question:
    get_and_display_response(question)
