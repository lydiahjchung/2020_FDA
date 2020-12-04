import sys
import json
import plotly
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pfo.preprocess import find_all, get_value, save_pfos
from pfo.allocation import Asset, Portfolio
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

def create_plot(feature):
    if feature == 'Bar':
        N = 40
        x = np.linspace(0, 1, N)
        y = np.random.randn(N)
        df = pd.DataFrame({'x': x, 'y': y})

        data = [
            go.Bar(
                x=df['x'],
                y=df['y']
            )
        ]
    
    else:
        N = 1000
        random_x = np.random.randn(N)
        random_y = np.random.randn(N)

        data = [
            go.Scatter(
                x = random_x,
                y = random_y,
                mode = 'markers'
            )
        ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bar', methods=['GET', 'POST'])
def change_features():
    print('working', file=sys.stderr)
    feature = request.args['selected']
    graphJSON = create_plot(feature)
    return graphJSON

@app.route('/post', methods=['POST'])
def post():
    # Allocating PFOS
    total = request.get_data(as_text=True).split('&')

    start = get_value(total[find_all(total, "start")[0]])
    end = get_value(total[find_all(total, "end")[0]])
    initial = get_value(total[find_all(total, "initial")[0]])
    rebalancing = get_value(total[find_all(total, "rebalancing")[0]])
    print(start, end, initial, rebalancing, file=sys.stderr)

    pfo_idx = find_all(total, "pfo_name")
    pfos = save_pfos(total, pfo_idx)
    print(pfos, file=sys.stderr)

    feature='Bar'
    bar = create_plot(feature)

    return render_template('result.html', result=pfos, plot=bar)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)

