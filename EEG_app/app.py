from flask import Flask, request, render_template
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow import keras

import numpy as np
import pandas as pd

import base64
import collections
import io
import pickle

app = Flask(__name__)


@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv('/Users/jessiexiong/Desktop/test/test.csv')
    x_test = data.drop('Label', axis=1).copy()

    reconstructed_model = keras.models.load_model("modelbci_0225")

    x_pred = np.array(list(map(lambda x: np.argmax(x), reconstructed_model.predict(x_test))))
    counter = collections.Counter(x_pred)

    fig = Figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')

    # get the data for the pie chart
    labels, data = [], []
    for item in counter.most_common():
        if item[0] == 2:
            labels.append("Concentrated")
        elif item[0] == 0:
            labels.append("Relaxed")
        data.append(item[1])

    colors = ['#DD7596', '#8EB897']

    ax.pie(data, labels=labels, colors = colors, autopct='%1.0f%%')

    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()

    return render_template('index.html', chart_url=chart_url)

if __name__ == "__main__":
    app.run(debug=False)