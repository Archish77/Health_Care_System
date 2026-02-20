from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("heart_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.json
        print("Received Data:", data)

        missing = [f for f in features if f not in data or data[f] == ""]
        if missing:
            return jsonify({
                "prediction": f"Missing fields: {', '.join(missing)}",
                "confidence": 0
            })

        input_values = [int(data[f]) for f in features]

        input_df = pd.DataFrame([input_values], columns=features)

       
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        confidence = probability[prediction]

        result = (
            "Heart Disease Detected"
            if prediction == 1
            else "No Heart Disease"
        )

        return jsonify({
            "prediction": result,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "prediction": "Server Error",
            "confidence": 0
        })

if __name__ == "__main__":
    app.run(debug=True)