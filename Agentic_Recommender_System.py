# basic imports
import streamlit as st
import pandas as pd
from collections import Counter
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# configuration
GEMINI_API_KEY = st.secrets["YOUR-GEMININ_API_KEY"]
SENDER_EMAIL = st.secrets["sender_email"]
SENDER_PASSWORD = st.secrets["sender_password"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


# Page Config
st.set_page_config(page_title="Agentic AI Recommender", layout="centered")
st.title(" Agentic Product Recommendation System")
st.caption("ML + RAG + Gemini LLM + Email Automation")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("ecommerce_data.csv")

df = load_data()
# normalize column names
df.columns = df.columns.str.strip().str.lower()

# RAG: Build Knowledge Base from Purchase History
def build_rag_corpus(customer_id):
    user_data = df[df["customerid"] == customer_id]
    if user_data.empty:
        return ""
    documents = user_data["description"].dropna().astype(str).tolist()
    return " ".join(documents)

# ML Recommendation Engine
def ml_recommender(customer_id, top_n=5):
    user_data = df[df["customerid"] == customer_id]
    if user_data.empty:
        return []
    products = user_data["description"].dropna().tolist()
    return [p[0] for p in Counter(products).most_common(top_n)]

# Gemini LLM with RAG
def generate_gemini_message(customer_id, recommendations, rag_context):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")

    prompt = f"""
You are an intelligent AI marketing assistant.

Customer ID: {customer_id}

Customer purchase history context:
{rag_context}

Recommended products:
{", ".join(recommendations)}

Generate a friendly, personalized product recommendation message
that sounds human, persuasive, and helpful.
"""

    response = model.generate_content(prompt)
    return response.text

# Email Sending Function
def send_email(receiver_email, message_body):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = receiver_email
        msg["Subject"] = "Personalized Product Recommendations"
        msg.attach(MIMEText(message_body, "plain"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        return True, "Email sent successfully"
    except Exception as e:
        return False, f"Email failed: {e}"

# AGENT CONTROLLER :Agentic AI Core
def agentic_pipeline(customer_id, email):
    # Step 1: RAG context retrieval
    rag_context = build_rag_corpus(customer_id)

    if not rag_context:
        return "No purchase history found for this customer.", False, "No history"

    # Step 2: ML recommendation
    recommendations = ml_recommender(customer_id)

    # Step 3: LLM personalization (Gemini)
    final_message = generate_gemini_message(
        customer_id, recommendations, rag_context
    )

    # Step 4: Email sending
    ok, email_status = send_email(email, final_message)
    return final_message, ok, email_status

# UI
st.subheader(" User Input")

customer_id = st.number_input("Customer ID", min_value=1, step=1)
email = st.text_input("Customer Email Address")

if st.button("Run Agentic Recommendation"):
    if not email:
        st.warning("Please enter email address")
    else:
        with st.spinner("Agent is working..."):
            message, ok, email_status = agentic_pipeline(customer_id, email)
            
        if ok:
            st.success("✅ Recommendation generated & email sent!")
        else:
            st.error("❌ Email not sent")
            st.info(email_status)
            st.text(message)

# Explanation
with st.expander("Agentic AI Workflow Explanation"):
    st.markdown("""
**1. Retrieval (RAG)**  
Customer purchase history is retrieved and used as context.

**2. ML Recommendation**  
Frequently purchased items are identified.

**3. Gemini LLM**  
Generates personalized messages using retrieved context.

**4. Agentic Control**  
The agent autonomously coordinates all steps.

**5. Email Automation**  
Final output is delivered to the customer via email.
""")

st.markdown("---")
st.caption("Agentic AI Project | ML + GenAI + RAG + Automation")