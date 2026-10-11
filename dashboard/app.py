
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Real-Time Financial Fraud Detection")
st.caption("AI-powered transaction risk assessment | Educational demonstration")

st.info(
    "This system supports fraud analysis. Predictions are estimates, "
    "not proof of fraud or a substitute for human review."
)

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
    st.success("Fraud detection API is connected.")
except requests.RequestException:
    st.error(
        "Cannot connect to the API. Start it in your first terminal "
        "using: python -m uvicorn api.main:app --reload"
    )
    st.stop()

st.subheader("Transaction details")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction step", min_value=1, value=1, step=1
        )
        amount = st.number_input(
            "Transaction amount", min_value=0.0, value=181.0
        )
        transaction_type = st.selectbox(
            "Transaction type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
        )

    with col2:
        oldbalanceOrg = st.number_input(
            "Origin balance before", min_value=0.0, value=181.0
        )
        newbalanceOrig = st.number_input(
            "Origin balance after", min_value=0.0, value=0.0
        )
        oldbalanceDest = st.number_input(
            "Destination balance before", min_value=0.0, value=0.0
        )
        newbalanceDest = st.number_input(
            "Destination balance after", min_value=0.0, value=0.0
        )

    submitted = st.form_submit_button(
        "Analyze transaction", type="primary", use_container_width=True
    )

if submitted:
    transaction = {
        "step": int(step),
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
        "type": transaction_type,
    }

    try:
        with st.spinner("Analyzing transaction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=transaction,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

        st.subheader("Prediction result")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric(
            "Fraud probability",
            f"{result['fraud_probability'] * 100:.2f}%",
        )
        metric2.metric("Prediction", str(result["prediction"]))
        metric3.metric("Risk level", result["risk_level"])

        if result["prediction"] == 1:
            st.error("Potential fraud detected. Send for human review.")
        else:
            st.success("No fraud detected at the current threshold.")

        st.caption(
            f"Decision threshold: {result['threshold']:.2f}"
        )
        st.json(result)

    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
