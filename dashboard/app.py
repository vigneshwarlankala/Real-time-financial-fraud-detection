
import os
import sqlite3

import requests
import streamlit as st

try:
    from database import (
        initialize_database,
        save_alert,
        get_alerts,
        update_alert,
    )
except ModuleNotFoundError:
    from dashboard.database import (
        initialize_database,
        save_alert,
        get_alerts,
        update_alert,
    )


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "https://fraud-detection-api-er42.onrender.com",
).rstrip("/")

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

initialize_database()


# --------------------------------------------------
# Dashboard styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.9);
    }

    [data-testid="stMetric"] {
        background: #151f32;
        border: 1px solid #293750;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stForm"] {
        background: #111b2d;
        border: 1px solid #293750;
        padding: 22px;
        border-radius: 14px;
    }

    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554, #102b3f);
        border: 1px solid #28446a;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: 2.1rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #cbd5e1;
        margin-bottom: 0;
    }

    .section-title {
        color: #f1f5f9;
        font-size: 1.2rem;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 9px;
        min-height: 46px;
        font-weight: 600;
    }

    hr {
        border-color: #293750;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Financial Fraud Detection</h1>
        <p>
            Real-time transaction risk assessment powered by machine learning.
            Analyze transactions, review alerts, and manage cases.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("AI/ML CAPSTONE PROJECT | TRANSACTION RISK ANALYTICS")

st.info(
    "This system provides educational fraud-risk estimates. "
    "A model prediction is not proof that a transaction is fraudulent."
)


# --------------------------------------------------
# API health status
# --------------------------------------------------

st.markdown(
    '<div class="section-title">System Overview</div>',
    unsafe_allow_html=True,
)

api_healthy = False

try:
    with st.spinner("Checking prediction API..."):
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=60,
        )
        health_response.raise_for_status()
        health_data = health_response.json()

    api_healthy = (
        health_data.get("status") == "healthy"
        and health_data.get("model_loaded") is True
    )

    if api_healthy:
        st.success("● Prediction API connected — model loaded")
    else:
        st.warning(
            f"API responded, but its health needs attention: {health_data}"
        )

except requests.RequestException as exc:
    st.error(f"Prediction API connection failed: {exc}")

except ValueError as exc:
    st.error(f"Invalid API health response: {exc}")

st.caption(f"API endpoint: {API_URL}")

st.divider()


# --------------------------------------------------
# Alert summary metrics
# --------------------------------------------------

alerts = get_alerts()

total_alerts = len(alerts)

high_risk_alerts = sum(
    1 for alert in alerts
    if alert["risk_level"].upper() == "HIGH"
)

open_cases = sum(
    1 for alert in alerts
    if alert["case_status"] == "Open"
)

investigating_cases = sum(
    1 for alert in alerts
    if alert["case_status"] == "Investigating"
)

st.markdown(
    '<div class="section-title">Alert Summary</div>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Total Alerts", total_alerts)
metric2.metric("High-Risk Alerts", high_risk_alerts)
metric3.metric("Open Cases", open_cases)
metric4.metric("Investigating", investigating_cases)

st.divider()


# --------------------------------------------------
# Transaction input form
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Analyze a Transaction</div>',
    unsafe_allow_html=True,
)

st.write(
    "Enter transaction details below. The model will estimate "
    "the risk using the supplied values."
)

with st.form("transaction_form"):
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Transaction Details**")

        step = st.number_input(
            "Transaction step",
            min_value=1,
            value=1,
            step=1,
            help="Time step associated with the transaction in the dataset.",
        )

        transaction_type = st.selectbox(
            "Transaction type",
            options=[
                "TRANSFER",
                "CASH_OUT",
                "PAYMENT",
                "CASH_IN",
                "DEBIT",
            ],
        )

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=181.0,
            step=1.0,
            format="%.2f",
        )

    with right:
        st.markdown("**Account Balances**")

        oldbalanceOrg = st.number_input(
            "Origin balance before transaction",
            min_value=0.0,
            value=181.0,
            step=1.0,
            format="%.2f",
        )

        newbalanceOrig = st.number_input(
            "Origin balance after transaction",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
        )

        oldbalanceDest = st.number_input(
            "Destination balance before transaction",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
        )

        newbalanceDest = st.number_input(
            "Destination balance after transaction",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Predict and save alert
# --------------------------------------------------

if submitted:
    transaction = {
        "step": int(step),
        "type": transaction_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest),
    }

    try:
        with st.spinner(
            "Analyzing transaction with the machine learning model..."
        ):
            response = requests.post(
                f"{API_URL}/predict",
                json=transaction,
                timeout=60,
            )

        response.raise_for_status()
        result = response.json()

        probability = result.get("fraud_probability")
        prediction = result.get("prediction")

        risk = str(
            result.get(
                "risk_level",
                result.get("risk", "UNKNOWN"),
            )
        ).upper()

        if probability is None:
            raise ValueError(
                "The API response did not contain a fraud probability."
            )

        probability = float(probability)

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Fraud probability must be between 0 and 1."
            )

        if prediction not in (0, 1):
            raise ValueError(
                f"Unexpected prediction value: {prediction}"
            )

        prediction = int(prediction)

        if not risk or risk == "UNKNOWN":
            raise ValueError(
                "The API did not return a valid risk classification."
            )

        # Persist the prediction in SQLite.
        alert_id = save_alert(
            transaction_type=transaction_type,
            amount=float(amount),
            fraud_probability=probability,
            prediction=prediction,
            risk_level=risk,
        )

        st.session_state["last_alert_id"] = alert_id

        st.success(
            f"Prediction completed and saved as Alert #{alert_id}."
        )

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True,
        )

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "Estimated fraud probability",
            f"{probability * 100:.2f}%",
        )

        result2.metric(
            "Risk classification",
            risk,
        )

        result3.metric(
            "Model prediction",
            str(prediction),
            help="1 means flagged; 0 means not flagged.",
        )

        st.progress(
            probability,
            text="Estimated fraud probability",
        )

        if prediction == 1:
            st.error(
                "⚠️ The model flagged this transaction for further review."
            )
        else:
            st.success(
                "The model did not flag this transaction as fraud."
            )

        if risk == "HIGH":
            st.error(
                "HIGH RISK — Review this transaction carefully."
            )
        elif risk == "LOW":
            st.success(
                "LOW RISK — The model assigned a low risk level."
            )
        else:
            st.info(f"Risk level returned by the API: {risk}")

        st.markdown("**Transaction Summary**")

        summary1, summary2, summary3 = st.columns(3)

        summary1.metric(
            "Transaction type",
            transaction_type,
        )

        summary2.metric(
            "Transaction amount",
            f"{amount:,.2f}",
        )

        summary3.metric(
            "Transaction step",
            str(step),
        )

        with st.expander("View transaction details and API response"):
            st.markdown("**Submitted transaction**")
            st.json(transaction)

            st.markdown("**API response**")
            st.json(result)

        st.caption(
            "A high-risk prediction is a screening signal, not a confirmed "
            "fraud determination. Review additional evidence before deciding."
        )

    except requests.HTTPError as exc:
        st.error(f"The prediction API returned an HTTP error: {exc}")

        if exc.response is not None:
            st.code(exc.response.text)

    except requests.RequestException as exc:
        st.error(f"Could not contact the prediction API: {exc}")

    except (TypeError, ValueError) as exc:
        st.error(f"Could not interpret the prediction response: {exc}")

    except sqlite3.Error as exc:
        st.error(f"Could not save the alert to the database: {exc}")


# --------------------------------------------------
# Alert history
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">Alert History & Case Management</div>',
    unsafe_allow_html=True,
)

st.write(
    "Review previous predictions, update case status, "
    "and record analyst feedback."
)

alerts = get_alerts()

if not alerts:
    st.info(
        "No alerts have been saved yet. Analyze a transaction above "
        "to create your first alert."
    )

else:
    history_rows = []

    for alert in alerts:
        history_rows.append(
            {
                "Alert ID": alert["id"],
                "Created At (UTC)": alert["created_at"],
                "Type": alert["transaction_type"],
                "Amount": alert["amount"],
                "Fraud Probability": (
                    f'{alert["fraud_probability"] * 100:.2f}%'
                ),
                "Prediction": alert["prediction"],
                "Risk": alert["risk_level"],
                "Case Status": alert["case_status"],
                "Analyst Feedback": alert["analyst_feedback"],
            }
        )

    st.dataframe(
        history_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Update a Case**")

    alert_options = {
        (
            f'Alert #{alert["id"]} — {alert["transaction_type"]} — '
            f'{alert["risk_level"]} — {alert["amount"]:,.2f}'
        ): alert["id"]
        for alert in alerts
    }

    selected_label = st.selectbox(
        "Select an alert to review",
        options=list(alert_options.keys()),
    )

    selected_id = alert_options[selected_label]

    selected_alert = next(
        alert for alert in alerts
        if alert["id"] == selected_id
    )

    with st.form(f"case_form_{selected_id}"):
        st.caption(
            f'Probability: {selected_alert["fraud_probability"] * 100:.2f}% '
            f'| Prediction: {selected_alert["prediction"]} '
            f'| Current status: {selected_alert["case_status"]}'
        )

        status_options = [
            "Open",
            "Investigating",
            "Closed",
        ]

        feedback_options = [
            "Pending",
            "Confirmed Fraud",
            "Legitimate",
            "Inconclusive",
        ]

        current_status = selected_alert["case_status"]
        current_feedback = selected_alert["analyst_feedback"]

        status_index = (
            status_options.index(current_status)
            if current_status in status_options
            else 0
        )

        feedback_index = (
            feedback_options.index(current_feedback)
            if current_feedback in feedback_options
            else 0
        )

        new_status = st.selectbox(
            "Case status",
            options=status_options,
            index=status_index,
        )

        new_feedback = st.selectbox(
            "Analyst feedback",
            options=feedback_options,
            index=feedback_index,
        )

        new_notes = st.text_area(
            "Analyst notes",
            value=selected_alert["analyst_notes"] or "",
            placeholder="Record the reason for the review or decision.",
            max_chars=2000,
        )

        save_case = st.form_submit_button(
            "Save Case Update",
            type="primary",
            use_container_width=True,
        )

    if save_case:
        try:
            updated = update_alert(
                alert_id=selected_id,
                case_status=new_status,
                analyst_feedback=new_feedback,
                analyst_notes=new_notes,
            )

            if updated:
                st.success(
                    f"Alert #{selected_id} updated successfully."
                )
                st.rerun()
            else:
                st.warning(
                    "The selected alert could not be found."
                )

        except sqlite3.Error as exc:
            st.error(f"Could not update the case: {exc}")

    with st.expander(f"View notes for Alert #{selected_id}"):
        st.write(
            selected_alert["analyst_notes"] or "No notes recorded."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption("Real-Time Financial Fraud Detection Platform")

st.caption(
    "Educational demonstration only. Not intended for real-world "
    "financial decisions or production fraud prevention."
)
