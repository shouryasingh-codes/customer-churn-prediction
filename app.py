from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib

# --------------------------------------------------
# Load trained model (server start hote hi)
# --------------------------------------------------
model = joblib.load("persona_churn_model.pkl")

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# Input schema
# --------------------------------------------------
class Customer(BaseModel):
    tenure: int
    last_active_days: int
    usage_frequency: int
    session_duration: int
    complaints: int
    monthly_charges: float
    payment_missed: int
    products_used: int
    plan_type: str

# --------------------------------------------------
# Business logic helpers
# --------------------------------------------------
def segment(prob):
    if prob < 0.2:
        return "low risk"
    elif prob < 0.6:
        return "medium risk"
    else:
        return "high risk"

def recommend(risk):
    if risk == "low risk":
        return "upsell premium"
    elif risk == "medium risk":
        return "send reminder email"
    else:
        return "discount + call"

# --------------------------------------------------
# Home endpoint
# --------------------------------------------------
@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(customer: Customer):

    # 1. Input → DataFrame
    df = pd.DataFrame([customer.dict()])

    # 2. One-hot encode plan_type
    df = pd.get_dummies(df, columns=["plan_type"], drop_first=False)

    # 3. Align columns with training
    expected_cols = model.feature_names_in_
    df = df.reindex(columns=expected_cols, fill_value=0)

    # 4. Predict probability
    prob = model.predict_proba(df)[0][1]

    # 5. Risk + action
    risk = segment(prob)
    action = recommend(risk)

    # 6. Business override (revenue protection)
    if (
        risk == "low risk"
        and customer.monthly_charges > 900
        and customer.products_used == 1
    ):
        risk = "medium risk"
        action = "send reminder email"

    return {
        "churn_probability": round(float(prob), 4),
        "risk": risk,
        "recommended_action": action
    }

# --------------------------------------------------
# SIMPLE UI ENDPOINT
# --------------------------------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head>
        <title>Customer Churn Predictor</title>
        <style>
            body { font-family: Arial; min-height: 100vh; background: linear-gradient(135deg, #1f2933, #111827); display: flex; justify-content: center; align-items: flex-start;padding-top: 40px;}
           .box { background: white; padding: 20px; border-radius: 10px; max-width: 600px; box-shadow: 0 10px 25px rgba(0,0,0,0.15);}
           input, select { width: 100%; padding: 6px; margin: 5px 0; }
            button { padding: 10px; width: 100%; background: black; color: white; }
            pre { background: #eee; padding: 10px; }
        </style>
    </head>
    <body>
        <div class="box">
         <h1 style="
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            margin-bottom: 10px;
        ">
            Customer Churn Prediction System
        </h1>
            <h2>Customer Details</h2>
            <input id="tenure" placeholder="Tenure (months)">
            <input id="last_active_days" placeholder="Last Active Days">
            <input id="usage_frequency" placeholder="Usage Frequency / week">
            <input id="session_duration" placeholder="Session Duration (mins)">
            <input id="complaints" placeholder="Complaints">
            <input id="monthly_charges" placeholder="Monthly Charges">
            <input id="payment_missed" placeholder="Payment Missed (0 or 1)">
            <input id="products_used" placeholder="Products Used">

            <select id="plan_type">
                <option value="basic">Basic</option>
                <option value="standard">Standard</option>
                <option value="premium">Premium</option>
            </select>

            <button onclick="predict()">Predict</button>

            <h3>Result</h3>
            <pre id="result"></pre>
        </div>

        <script>
            function predict() {
                fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        tenure: Number(tenure.value),
                        last_active_days: Number(last_active_days.value),
                        usage_frequency: Number(usage_frequency.value),
                        session_duration: Number(session_duration.value),
                        complaints: Number(complaints.value),
                        monthly_charges: Number(monthly_charges.value),
                        payment_missed: Number(payment_missed.value),
                        products_used: Number(products_used.value),
                        plan_type: plan_type.value
                    })
                })
                .then(res => res.json())
                .then(data => {
                    document.getElementById("result").innerText =
`Customer Summary
----------------
Tenure           : ${tenure.value} months
Last Active Days : ${last_active_days.value}
Usage Frequency  : ${usage_frequency.value}/week
Session Duration : ${session_duration.value} mins
Complaints       : ${complaints.value}
Monthly Charges  : ₹${monthly_charges.value}
Payment Missed   : ${payment_missed.value == 1 ? "Yes" : "No"}
Products Used    : ${products_used.value}
Plan Type        : ${plan_type.value}

Prediction
-----------
Churn Probability : ${data.churn_probability}
Risk Level        : ${data.risk}
Recommended Action: ${data.recommended_action}`;
                });
            }
        </script>
    </body>
    </html>
    """
