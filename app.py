
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("iris_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        input_data = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
        )

        prediction = model.predict(input_data)
        predicted_species = label_encoder.inverse_transform(prediction)[0]

        return render_template(
            "index.html",
            prediction_text="Predicted Iris Flower Species: " + predicted_species
        )

    except:
        return render_template(
            "index.html",
            prediction_text="Error: Please enter valid numeric values."
        )

if __name__ == "__main__":
    app.run(debug=False)
