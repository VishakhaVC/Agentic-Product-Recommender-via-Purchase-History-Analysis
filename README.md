# Agentic Product Recommender via Purchase History Analysis
This project implements an agentic AI-powered product recommendation system that analyzes customer purchase history, applies a machine learning recommender, enriches context using RAG (Retrieval-Augmented Generation), and uses a Gemini LLM to generate personalized product recommendation messages. The system delivers results through a Streamlit dashboard and automated email notifications, showcasing a complete autonomous AI workflow for e-commerce personalization.

# Problem Statement
Traditional recommender systems rely mostly on static ML models and lack intelligent, personalized communication. They also require manual coordination between data processing, recommendation, and messaging. This project solves that by building an agentic system that autonomously:
  - Retrieves user data
  - Cleans and validates it
  - Generates recommendations using ML
  - Uses an LLM to create human-like messages
  - Delivers results via email

# Solution Overview
The system follows a multi-agent pipeline:
  - Uses Pandas and NumPy for data processing
  - Applies a purchase-history–based ML recommender
  - Uses RAG to provide contextual understanding
  - Uses Gemini LLM to generate personalized recommendation messages

# Dataset: https://www.kaggle.com/datasets/carrie1/ecommerce-data

# System Architecture
  - User Interaction Agent – Accepts customer ID and email via Streamlit UI
  - Data Retrieval Agent (RAG) – Fetches customer purchase history as context
  - Data Cleaning Agent – Validates, cleans, and normalizes dataset
  - ML Recommendation Agent – Recommends products using frequency/similarity logic
  - LLM Personalization Agent (Gemini) – Generates human-like recommendation messages
  - Automation Agent – Sends personalized emails and displays output

# Tech Stack
  - Python – Core language
  - Pandas – Data loading, cleaning, analysis
  - NumPy – Numerical computations
  - Scikit-learn – TF-IDF & similarity-based recommendations
  - Streamlit – Interactive web dashboard
  - Gemini LLM (Google GenAI) – Personalized message generation
  - RAG (Retrieval-Augmented Generation) – Context from purchase history
  - SMTP – Automated email sending
  - Matplotlib – EDA visualizations
  - CSV Dataset – E-commerce transactional data

# Use Cases
  - Personalized marketing in e-commerce platforms
  - AI-driven customer engagement systems
  - Educational demonstration of Agentic AI + GenAI
  - Proof-of-concept for intelligent recommendation engines

# Limitations
  - Recommendation logic is currently frequency/similarity-based
  - Depends on historical purchase data
  - No real-time user behavior tracking yet
  - Email credentials must be handled securely

# Future Enhancements
  - Deep learning–based recommenders
  - Real-time behavior tracking
  - Sentiment-based personalization
  - Multi-channel delivery (SMS, chatbot, push notifications)
  - Cloud deployment and scalability
  - Displays results in Streamlit and sends them via email automation

# How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Add your Gemini API key in Agentic_Recommender_System.py
3. Add sender email address and password in both files
4. Run: streamlit run app.py
