from flask import Flask, request, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow import keras

import numpy as np
import pandas as pd

import base64
import collections
import glob
import io
import os
import pickle

from EEG_generate_training_matrix import gen_training_matrix

app = Flask(__name__)

@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        filepath = request.form.get("filepath")
        if filepath == "f":
            # relaxed
            filepath = '/Users/jessiexiong/Desktop/test_G/test-3.csv'
        data=pd.read_csv(filepath)
        x_test = data.drop('Label', axis=1).copy()

        chart_url = model_predict_to_pie(x_test)

        return render_template('index.html', chart_url=chart_url)

    except Exception as err:
        warning = str(err)+" Could not run, try again?"
        return render_template('index.html', error=warning)


@app.route('/predict2',methods=['POST'])
def predict2():
    #
    try:
        list_of_files = glob.glob('/Users/jessiexiong/Documents/OpenBCI_GUI/Recordings/*/*.csv')  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)

        # df = pd.read_csv("/Users/jessiexiong/Desktop/test/aine-relaxed-1.csv", names=["Header"])
        df = pd.read_csv(latest_file, names=["Header"])

        df = df['Header'].str.split('\t', expand=True)
        df = df.iloc[:, 0:5]

        data = gen_training_matrix(df)

        chart_url = model_predict_to_pie(data)
        return render_template('index.html', chart_url=chart_url)

    except Exception as err:
        warning = str(err) + " Could not run, try again?"
        return render_template('index.html', error=warning)


def model_predict_to_pie(X_pred):
    """Run model.predict on data and convert data to pie chart"""
    reconstructed_model = keras.models.load_model("../ML model/Run_4/modelbci_demo3")

    x_pred = np.round(reconstructed_model.predict(X_pred))

    fig = Figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    # get the data for the pie chart
    labels, data, colors = ["Concentrated", "Relaxed"], [np.count_nonzero(x_pred == 1.0),
                                                         np.count_nonzero(x_pred == 0.0)], ['#B1D4E0', '#4FA64F']

    ax.pie(data, labels=labels, colors=colors, autopct='%1.1f%%')

    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    return chart_url


if __name__ == "__main__":
    app.run(debug=False)
