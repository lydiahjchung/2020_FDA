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

portfolio = []

def create_plot(feature, portfolio):
    if feature == 'Drawdown':
        data = []
        for i in range(len(portfolio)):
            date = portfolio[i].get_date()
            result = portfolio[i].get_backtest_result()
            data.append(
                go.Scatter(
                    x = date, 
                    y = result.loc[:, 'Total Drawdown'],
                    mode = 'lines',
                    name = portfolio[i].get_name()
                )
            )
        
        layout = go.Layout(
            title = 'Portfolio Drawdown Comparison',
        )
    
    elif feature == 'Cumreturn':
        data = []
        for i in range(len(portfolio)):
            date = portfolio[i].get_date()
            result = portfolio[i].get_backtest_result()
            data.append(
                go.Scatter(
                    x = date, 
                    y = result.loc[:, 'Total Cum. Return'],
                    mode = 'lines',
                    name = portfolio[i].get_name()
                )
            )
        
        layout = {}

    else: # Balance
        X = [[0,1,2], [0,1,2]]
        Y = [[3,6,9], [8,6,4]]
        
        data = []
        for i in range(len(X)):
            data.append(
                go.Scatter(
                    x = X[i], 
                    y = Y[i],
                    mode='lines'
                )
            )

        layout = {}

    entire = [data, layout]
    graphJSON = json.dumps(entire, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['GET', 'POST'])
def change_features():
    print('working', file=sys.stderr)
    feature = request.args['selected']
    print("check", portfolio, file=sys.stderr)
    graphJSON = create_plot(feature, portfolio)
    return graphJSON

@app.route('/post', methods=['POST'])
def post():
    # Allocating PFOS
    total = request.get_data(as_text=True).split('&')

    start = get_value(total[find_all(total, "start")[0]])
    end = get_value(total[find_all(total, "end")[0]])
    initial = int(get_value(total[find_all(total, "initial")[0]]))
    rebalancing = int(get_value(total[find_all(total, "rebalancing")[0]]))

    pfo_idx = find_all(total, "pfo_name")
    pfos = save_pfos(total, pfo_idx)

    for name in pfos.keys():
        assets = []
        data = pfos[name]

        for i in range(len(data['tickers'])):
            assets.append(Asset(data['tickers'][i], data['names'][i], start, end))
        
        temp = Portfolio(name, assets, data['weights'], data['leverages'], initial, rebalancing)
        temp.backtest()
        temp.backtest_result()
        portfolio.append(temp)

    feature='Drawdown'
    bar = create_plot(feature, portfolio)

    return render_template('result.html', result=pfos, plot=bar)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)

