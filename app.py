from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
model = pickle.load(open("model/model.pkl", "rb"))
sc = pickle.load(open("model/scaler.pkl", "rb"))

@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    #init_features = [[float(x) for x in request.form.values()]]
    init_features = pd.DataFrame(request.form, index=[0])
    init_features_sc = sc.transform(init_features)
    print(init_features_sc)
    prediction = model.predict(init_features_sc)
    return render_template("index.html", Prediction=f"The predicted :{prediction[0]}")

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=os.environ.get("PORT", 5000))