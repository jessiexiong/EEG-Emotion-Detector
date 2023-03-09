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
            filepath = '/Users/jessiexiong/Desktop/test/preprocessed-5.csv'
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
        x_test = np.delete(data, -1, axis=1).copy()

        chart_url = model_predict_to_pie(x_test)
        return render_template('index.html', chart_url=chart_url)

    except Exception as err:
        warning = str(err) + " Could not run, try again?"
        return render_template('index.html', error=warning)


def model_predict_to_pie(x_test):
    """Run model.predict on data and convert data to pie chart"""
    reconstructed_model = keras.models.load_model("modelbci_0225")

    x_pred = np.array(list(map(lambda x: np.argmax(x), reconstructed_model.predict(x_test))))

    # convert data to pie chart
    counter = collections.Counter(x_pred)
    fig = Figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    # get the data for the pie chart
    labels, data, colors = [], [], [0,0]
    for item in counter.most_common():
        if item[0] == 2:
            labels.append("Concentrated")
            colors[0]='#B1D4E0'
        elif item[0] == 0:
            labels.append("Relaxed")
            colors[1]='#4FA64F'
        data.append(item[1])

    ax.pie(data, labels=labels, colors=colors, autopct='%1.1f%%')

    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    return chart_url

if __name__ == "__main__":
    app.run(debug=True)