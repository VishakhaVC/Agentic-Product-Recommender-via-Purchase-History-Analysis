import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SENDER_EMAIL = "sender_email"
SENDER_PASSWORD = "sender_password"  # Gmail App Password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# page configuration
st.set_page_config(page_title="Agentic Product Recommender", layout="wide")
st.title("🧠 Agentic Product Recommendation System")
st.caption("Agentic AI | ML | RAG | EDA | Dashboard")

# dataset upload
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload E-commerce CSV Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload the dataset to proceed.")
    st.stop()

# load dataset
try:
    df = pd.read_csv(uploaded_file, encoding="utf-8", header=None)
except UnicodeDecodeError:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", header=None)

# assign column names
df.columns = [
    "invoiceno",
    "stockcode",
    "description",
    "quantity",
    "invoicedate",
    "unitprice",
    "customerid",
    "country"
]

# column cleaning
df.columns = (
    df.columns
    .str.encode("ascii", "ignore")
    .str.decode("ascii")
    .str.strip()
    .str.lower()
    .str.replace(" ", "")
    .str.replace("_", "")
)

#column mapping
final_column_map = {}

for col in df.columns:
    if "customer" in col:
        final_column_map[col] = "customerid"
    elif "description" in col or "product" in col:
        final_column_map[col] = "description"
    elif "quantity" in col or col == "qty":
        final_column_map[col] = "quantity"
    elif "unitprice" in col or "price" in col:
        final_column_map[col] = "unitprice"

df = df.rename(columns=final_column_map)

# data validation
required_columns = ["customerid", "description", "quantity", "unitprice"]
missing_cols = [c for c in required_columns if c not in df.columns]

if missing_cols:
    st.error("Dataset columns could not be mapped correctly.")
    st.write("Detected columns after cleaning:", df.columns.tolist())
    st.stop()

st.success("✅ Dataset loaded & validated successfully!")

# data cleaning
df = df.dropna(subset=["customerid", "description"])
df = df[df["quantity"] > 0]
df = df[df["unitprice"] > 0]

df["revenue"] = df["quantity"] * df["unitprice"]

# dataset overview
st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Unique Customers", df["customerid"].nunique())
col3.metric("Unique Products", df["description"].nunique())

with st.expander("Preview Dataset"):
    st.dataframe(df.head())

# eda
st.subheader("📈 Exploratory Data Analysis")

# Top Products Chart
top_products = df["description"].value_counts().head(10)

fig1, ax1 = plt.subplots()
top_products.plot(kind="bar", ax=ax1)
ax1.set_title("Top 10 Most Sold Products")
ax1.set_ylabel("Purchase Count")
ax1.set_xlabel("Product")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig1)

# Revenue Distribution
filtered_revenue = df[df["revenue"] < df["revenue"].quantile(0.99)]

fig2, ax2 = plt.subplots()
ax2.hist(filtered_revenue["revenue"], bins=50)
ax2.set_title("Revenue Distribution")
ax2.set_xlabel("Revenue")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)

# top product dashboard
st.subheader("🏆 Top Products Dashboard")

d1, d2 = st.columns(2)

with d1:
    st.write("🔝 Top Products by Quantity")
    st.dataframe(
        df.groupby("description")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

with d2:
    st.write("💰 Top Products by Revenue")
    st.dataframe(
        df.groupby("description")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

# agentic ai recommendation
st.subheader("🤖 Agentic Recommendation Engine")
# added
st.markdown("📧 Send Recommendation via Email")
#

customer_id = st.number_input(
    "Enter Customer ID",
    min_value=1,
    step=1,
    key="agent_customer_id"
)

email = st.text_input(
    "Enter Customer Email Address",
    key="agent_email"
)



def retrieve_context(customer_id):
    user_data = df[df["customerid"] == customer_id]
    if user_data.empty:
        return ""
    return ", ".join(user_data["description"].head(20).tolist())

def recommend_products(customer_id, top_n=5):
    user_data = df[df["customerid"] == customer_id]
    if user_data.empty:
        return []
    products = user_data["description"].tolist()
    return [p[0] for p in Counter(products).most_common(top_n)]

def generate_message(customer_id, recommendations, context):
    return f"""
Hello Customer {int(customer_id)} 👋,

Based on your purchase history such as:
{context}

We recommend the following products for you:
👉 {", ".join(recommendations)}

Happy Shopping 😊
"""

def agent_pipeline(customer_id):
    context = retrieve_context(customer_id)
    if not context:
        return "No purchase history found for this customer."
    recommendations = recommend_products(customer_id)
    return generate_message(customer_id, recommendations, context)

# email sending function
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


# ML-based recommendation logic
if st.button("Generate & Send Recommendation"):
    if not email:
        st.warning("Please enter an email address")
    else:
        with st.spinner("Agent is reasoning..."):
            output = agent_pipeline(customer_id)

        ok, status = send_email(email, output)

        if ok:
            st.success("✅ Recommendation generated & email sent!")
        else:
            st.error("❌ Email not sent")
        st.info(status)
        st.text(output)
#

# additional
st.subheader("✨ Additional Insights")

# Top Customers
st.write("💳 Top Customers by Revenue")
top_customers = (
    df.groupby("customerid")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.dataframe(top_customers)

# Product Popularity
st.write("📦 Product Popularity Distribution")
fig3, ax3 = plt.subplots()
df["description"].value_counts().head(15).plot(kind="barh", ax=ax3)
ax3.set_title("Top 15 Products by Popularity")
st.pyplot(fig3)


# explanation section
with st.expander("📘 Project Explanation"):
    st.markdown("""
**1. Manual Dataset Upload**  
Users upload real-world e-commerce datasets.

**2. Robust Data Validation**  
Automatic column cleaning and semantic mapping ensure reliability.

**3. Exploratory Data Analysis (EDA)**  
Visual insights into products, revenue, and customers.

**4. Agentic AI System**  
An intelligent agent retrieves context, reasons, and generates output.

**5. ML Recommendation Logic**  
Frequency-based recommendation using purchase history.

**6. GenAI Concept**  
Human-like personalized recommendation messages.
""")

st.markdown("---")
st.caption("FINAL Agentic AI Project | PGDM( AI & DS )")
