
import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Real-Time Financial Fraud Detection")
st.caption(
    "AI-powered transaction risk assessment | Educational demonstration"
)

st.info(
    "This system supports fraud analysis. Predictions are estimates, "
    "not proof that a transaction is fraudulent."
)

# --------------------------------------------------
# API connection status
# --------------------------------------------------

st.subheader("System Status")

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()

    st.success("Prediction API is connected and healthy.")

except requests.RequestException:
    st.error(
        "Prediction API is unavailable. Start the FastAPI server "
        "or verify the API_URL configuration."
    )


# --------------------------------------------------
# Transaction input form
# --------------------------------------------------

st.subheader("Analyze a Transaction")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction step",
            min_value=1,
            value=1,
            step=1,
        )

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=181.0,
            step=1.0,
        )

        transaction_type = st.selectbox(
            "Transaction type",
            options=[
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER",
            ],
            index=4,
        )

    with col2:
        oldbalanceOrg = st.number_input(
            "Origin balance before transaction",
            min_value=0.0,
            value=181.0,
            step=1.0,
        )

        newbalanceOrig = st.number_input(
            "Origin balance after transaction",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

        oldbalanceDest = st.number_input(
            "Destination balance before transaction",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

        newbalanceDest = st.number_input(
            "Destination balance after transaction",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Send transaction to FastAPI
# --------------------------------------------------

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

        st.subheader("Prediction Results")

        probability = result.get("fraud_probability")
        prediction = result.get("prediction")
        risk = result.get("risk_level", result.get("risk", "UNKNOWN"))

        if probability is not None:
            st.metric(
                "Estimated fraud probability",
                f"{float(probability) * 100:.2f}%",
            )

            st.progress(
                max(0.0, min(1.0, float(probability)))
            )

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("Model prediction: Potential fraud")
            elif prediction == 0:
                st.success("Model prediction: Not flagged as fraud")
            else:
                st.warning(f"Model prediction: {prediction}")

        with col2:
            if str(risk).upper() == "HIGH":
                st.error(f"Risk level: {risk}")
            elif str(risk).upper() == "LOW":
                st.success(f"Risk level: {risk}")
            else:
                st.write(f"Risk level: {risk}")

        st.caption(
            "This is a machine-learning estimate based on the supplied "
            "transaction fields. It must not be treated as a confirmed "
            "fraud determination."
        )

        with st.expander("View submitted transaction and API response"):
            st.json(
                {
                    "transaction": transaction,
                    "prediction_response": result,
                }
            )

    except requests.HTTPError as exc:
        st.error(
            f"The API rejected the request or returned an error: {exc}"
        )
        if exc.response is not None:
            st.code(exc.response.text)

    except requests.RequestException as exc:
        st.error(f"Could not contact the prediction API: {exc}")

    except (TypeError, ValueError) as exc:
        st.error(f"Could not interpret the prediction response: {exc}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(
    "Educational demonstration only. Not intended for real financial "
    "decisions or production fraud prevention."
)
