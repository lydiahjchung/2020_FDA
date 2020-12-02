from flask import Flask, render_template, request

import plotly
import plotly.graph_objs as go

import pandas as pd
import sys
import numpy as np

from pfo.allocation import Asset, Portfolio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post', methods=['POST'])
def post():
    start = request.form.get("start")
    end = request.form.get("end")
    tickers = [t for t in request.form.getlist("asset[]") if t]
    names = [n for n in request.form.getlist("name[]") if n]
    weights = [int(w) for w in request.form.getlist("weight[]") if w]
    leverages = [int(l) for l in request.form.getlist("leverage[]") if l]
    
    lens = [len(tickers), len(weights), len(leverages)]
    assert len(set(lens)) == 1

    assets = []
    for i in range(len(tickers)):
        asset = Asset(tickers[i], names[i], start, end)
        assets.append(asset)
    print(len(assets), assets[0].get_name(), assets[1].get_name(), file=sys.stderr)

    return render_template("result.html", assets=assets, weights=weights, leverages=leverages)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)

