import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
regmodel = pickle.load(open('boston_housing_model.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    prediction = regmodel.predict(new_data)
    print(prediction[0])
    return jsonify(prediction[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    prediction = regmodel.predict(final_input)
    print(prediction[0])
    return render_template('home.html', prediction_text='Predicted Price: ${:,.4f}'.format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)

