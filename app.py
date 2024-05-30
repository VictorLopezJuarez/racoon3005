import os
import streamlit as st
from google.cloud import aiplatform
from google.oauth2 import service_account
import json
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI

# Load JSON credentials from Streamlit secrets
credentials_info = st.secrets["google_application_credentials"]
credentials_dict = json.loads(credentials_info)

# Write the JSON credentials to a file
with open("gcp_credentials.json", "w") as f:
    json.dump(credentials_dict, f)

# Set the environment variable to the path of the credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"

# Initialize the Vertex AI client
client = aiplatform.gapic.PredictionServiceClient(
    credentials=service_account.Credentials.from_service_account_file("gcp_credentials.json")
)

# Define the retrieval function
def get_and_display_response(question):
    chain = RetrievalQA.from_chain_type(llm=VertexAI(model="text-bison@001"), chain_type="stuff")
    response = chain(question)
    st.write(response)

# Streamlit app
st.title("AI Chatbot")
question = st.text_input("Ask a question:")
if question:
    get_and_display_response(question)
