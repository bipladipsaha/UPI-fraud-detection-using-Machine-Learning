from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
from datetime import datetime
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from feature_engineering import haversine, CITY_COORDS



app = Flask(__name__)
app.secret_key = "secret123"

# ==================================================
# LOAD MODEL FILES
# ==================================================
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# ==================================================
# HOME → LOGIN
# ==================================================
@app.route("/overview")
def overview():
    return render_template("overview.html")
@app.route("/")
def home():
    return redirect(url_for("overview"))

# ==================================================
# LOGIN
# ==================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        upi = request.form["username"]
        password = request.form["password"]
        role = request.form["role"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE upi = ?", (upi,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            session["upi"] = upi
            session["role"] = role
            return redirect(url_for("check"))

        return render_template(
            "login.html",
            error="Invalid UPI ID or password"
        )

    return render_template("login.html")


# ==================================================
# FRAUD CHECK + HISTORY (NO CHART)
# ==================================================
@app.route("/check", methods=["GET", "POST"])
def check():

    result = probability = reason_text = None

    if request.method == "POST":
        # -------- RAW INPUTS --------
        amount = float(request.form["amount"])
        oldbalance = float(request.form["oldbalance"])
        newbalance = float(request.form["newbalance"])
        hour = int(request.form["hour"])
        txn_type = request.form["transaction_type"]
        user_city = request.form["user_city"]
        merchant_city = request.form["merchant_city"]
        device_type = request.form["device_type"]

        # -------- FEATURE ENGINEERING --------
        errorBalanceOrig = oldbalance - newbalance - amount
        is_night = 1 if 0 <= hour <= 5 else 0
        is_peak = 1 if 9 <= hour <= 17 else 0

        # Location features
        user_lat, user_lon = CITY_COORDS[user_city]
        merchant_lat, merchant_lon = CITY_COORDS[merchant_city]

        distance_km = haversine(user_lat, user_lon, merchant_lat, merchant_lon)
        same_city = 1 if user_city == merchant_city else 0

        # -------- MODEL INPUT --------
        data = {
            "amount": amount,
            "oldbalanceOrg": oldbalance,
            "newbalanceOrig": newbalance,
            "hour": hour,
            "errorBalanceOrig": errorBalanceOrig,
            "is_night": is_night,
            "is_peak": is_peak,
            "distance_km": distance_km,
            "same_city": same_city,
            "transaction_type_PAYMENT": 0,
            "transaction_type_TRANSFER": 0,
            "device_type_Android": 0,
            "device_type_iOS": 0,
            "device_type_Web": 0,
        }

        data[f"transaction_type_{txn_type}"] = 1
        data[f"device_type_{device_type}"] = 1

        df = pd.DataFrame([data])

        # -------- ALIGN COLUMNS --------
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns]

        # -------- PREDICTION --------
        df_scaled = scaler.transform(df)
        prob = model.predict_proba(df_scaled)[0][1]
        pred = model.predict(df_scaled)[0]

        # -------- RULE LOGIC --------
        reasons = []

        if abs(errorBalanceOrig) > 1:
            reasons.append("Balance mismatch detected")
            pred = 1
            prob = max(prob, 0.95)

        if amount > 50000:
            reasons.append("High transaction amount")

        if is_night:
            reasons.append("Late-night transaction")

        if txn_type == "TRANSFER":
            reasons.append("High-risk transfer")

        if distance_km > 500:
            reasons.append("Transaction from distant location")
            prob = max(prob, 0.9)

        # -------- FINAL OUTPUT --------
        result = "🚨 FRAUD" if pred == 1 else "✅ SAFE"
        probability = round(prob * 100, 2)
        reason_text = ", ".join(reasons) if reasons else "Normal transaction behavior"
        decision = "BLOCK" if pred == 1 else "ALLOW"

        # -------- SAVE HISTORY --------
        log = pd.DataFrame([{
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "hour": hour,
            "type": txn_type,
            "risk": probability,
            "decision": decision,
            "result": result
        }])

        if os.path.exists("audit_log.csv"):
            log.to_csv("audit_log.csv", mode="a", header=False, index=False)
        else:
            log.to_csv("audit_log.csv", index=False)

    # -------- LOAD HISTORY --------
    if os.path.exists("audit_log.csv"):
        history = pd.read_csv("audit_log.csv").tail(10).to_dict(orient="records")
    else:
        history = []

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        reason=reason_text,
        history=history
    )
#//////////////////////////////////////////////
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        phone = request.form["phone"]
        upi = request.form["upi"]
        password = request.form["password"]

        hashed_pw = generate_password_hash(password)

        try:
            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (name, phone, upi, password) VALUES (?, ?, ?, ?)",
                (name, phone, upi, hashed_pw)
            )
            conn.commit()
            conn.close()

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            return render_template(
                "register.html",
                error="UPI ID already registered"
            )

    return render_template("register.html")

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
