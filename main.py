import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

model = joblib.load("model.pkl")


@app.route("/")
def home():
    """Render the login page."""
    return render_template("admin.html")


@app.route("/index")
def index():
    """Render the prediction form."""
    return render_template("index1.html")


@app.route("/adminval", methods=["POST", "GET"])
def adminval():
    """Validate admin login credentials."""
    if request.method == "POST":
        uname = request.form.get("username", "")
        upass = request.form.get("password", "")
        if uname == "admin" and upass == "admin123":
            return render_template("index1.html")
        else:
            return render_template("admin.html", msg="Invalid username or password.")
    return render_template("admin.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept transaction features and return ransomware prediction."""
    try:
        year = float(request.form.get("year", 0))
        day = float(request.form.get("day", 0))
        length = float(request.form.get("length", 0))
        weight = float(request.form.get("weight", 0))
        count = float(request.form.get("count", 0))
        looped = float(request.form.get("looped", 0))
        neighbors = float(request.form.get("neighbors", 0))
        income = float(request.form.get("income", 0))
    except ValueError:
        return render_template(
            "result.html",
            msg="Error",
            detail="Invalid input. Please enter numeric values.",
        )

    features = [year, day, length, weight, count, looped, neighbors, income]
    prediction = model.predict([features])
    label = prediction[0]

    if label == "white":
        detail = "This address shows no ransomware activity patterns."
    else:
        detail = f"This address matches patterns associated with the {label} ransomware family."

    return render_template("result.html", msg=label, detail=detail)


if __name__ == "__main__":
    app.run(debug=False, port=2000)
