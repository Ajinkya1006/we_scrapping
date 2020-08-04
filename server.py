import numpy as np
from flask import Flask, request, render_template, json, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('pune_rent_model_pickle.pkl', 'rb'))

with open('columns.json', 'r') as f:
    __data_columns = json.load(f)['data_columns']

data_columns = np.array(__data_columns, dtype="object")


def predict_price(bhk, area, condition, location):
    loc_index1 = np.where(data_columns == location)[0][0]
    loc_index2 = np.where(data_columns == condition)[0][0]
    X = np.zeros(len(data_columns))
    X[0] = bhk
    X[1] = area
    X[loc_index1] = 1
    X[loc_index2] = 1
    return round(model.predict([X])[0], 2)


@app.route("/home")
def home():
    return render_template('index.html')


@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        bhk = float(request.form['bhk'])
        area = float(request.form['area'])
        location = str(request.form.get('location'))
        condition = str(request.form.get('location'))
        output = predict_price(bhk, area, condition, location)
        return render_template('index.html', prediction_text='approx rent will be Rs{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
