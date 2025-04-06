from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "Iris Classification API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([data["features"]])  # Expecting a JSON with "features"
    prediction = model.predict(features)
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    return jsonify({"prediction": species_map[prediction[0]]})

if __name__ == "__main__":
    app.run(debug=True)