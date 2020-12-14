import sys
import json
import plotly
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pfo.preprocess import find_all, get_value, save_pfos, alloc_6040
from pfo.allocation import Asset, Portfolio_6040
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

portfolio = []
comparisons = []

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
            xaxis = dict(
                title = 'Date'
            ),
            yaxis = dict(
                title = 'Drawdown'
            )
        )
    
    elif feature == 'Cumreturn':
        data = []
        for i in range(len(portfolio)):
            date = portfolio[i].get_date()
            result = portfolio[i].get_backtest_result()
            print(result.loc[:, 'Total Cumulative Return'], file=sys.stderr)
            data.append(
                go.Scatter(
                    x = date, 
                    y = result.loc[:, 'Total Cumulative Return'],
                    mode = 'lines',
                    name = portfolio[i].get_name()
                )
            )
        
        layout = go.Layout(
            title = 'Portfolio Cumulative Return Comparison',
            xaxis = dict(
                title = 'Date'
            ),
            yaxis = dict(
                title = 'Cumulative Return'
            )
        )

    elif feature == 'Balance': 
        data = []
        for i in range(len(portfolio)):
            date = portfolio[i].get_date()
            result = portfolio[i].get_backtest_result()
            data.append(
                go.Scatter(
                    x = date, 
                    y = result.loc[:, 'Total Balance'],
                    mode = 'lines',
                    name = portfolio[i].get_name(),
                )
            )
        
        layout = go.Layout(
            title = 'Portfolio Balance Comparison',
            xaxis = dict(
                title = 'Date'
            ),
            yaxis = dict(
                title = 'Balance',
                type = 'log'
            )
        )

    elif feature == 'Monthly':
        data = []
        for i in range(len(portfolio)):
            date = portfolio[i].get_date()
            result = portfolio[i].get_monthly_result()
            data.append(
                go.Bar(
                    x = date, 
                    y = result.loc[:, 'Total Monthly Return'],
                    name = portfolio[i].get_name(),
                )
            )

        layout = go.Layout(
            title = 'Portfolio Monthly Return Comparison',
            xaxis = dict(
                title = 'Date'
            ),
            yaxis = dict(
                title = 'Return',
                type = 'linear'
            )
        )

    else: # Annual
        data = []
        for i in range(len(portfolio)):
            date = portfolio[i].get_date()
            result = portfolio[i].get_annual_result()
            data.append(
                go.Bar(
                    x = date, 
                    y = result.loc[:, 'Total Annual Return'],
                    name = portfolio[i].get_name(),
                )
            )

        layout = go.Layout(
            title = 'Portfolio Annual Return Comparison',
            xaxis = dict(
                title = 'Date'
            ),
            yaxis = dict(
                title = 'Return',
                type = 'linear'
            )
        )

    entire = [data, layout]
    graphJSON = json.dumps(entire, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/check', methods=['POST'])
def checkbox_value():
    nominal = request.form.getlist('nominal_bonds')
    global_bonds = request.form.getlist('global_bonds')
    corporate = request.form.getlist('corporate')
    global_equities = request.form.getlist('global_equities')
    emd_spreads = request.form.getlist('emd_spreads')
    commodities = request.form.getlist('commodities')
    
    tickers = [nominal, global_bonds, corporate, global_equities, emd_spreads, commodities]
    start = request.form.get('start')
    end = request.form.get('end')
    initial = int(request.form.get('initial'))
    rebalancing = int(request.form.get('rebalancing'))
    lookback = int(request.form.get('lookback'))

    # total = [tickers, start, end, initial, rebalancing, lookback]

    asset_6040, ratio_6040 = alloc_6040(tickers)
    pfo_6040 = Portfolio_6040('60/40', [Asset(each, '', start, end) for each in asset_6040], ratio_6040, initial, rebalancing)
    comparisons.append(pfo_6040)

    feature = 'Drawdown'
    bar = create_plot(feature, comparisons)
    bar = json.loads(bar)
    
    return render_template('result.html', plot=bar[0], layout=bar[1])

@app.route('/plot', methods=['GET', 'POST'])
def change_features():
    print('working', file=sys.stderr)
    feature = request.args['selected']
    print("check", comparisons, file=sys.stderr)
    graphJSON = create_plot(feature, comparisons)
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
        
        temp = Portfolio(name, assets, data['weights'], initial, rebalancing)
        temp.backtest()
        temp.backtest_result()
        portfolio.append(temp)

    feature='Drawdown'
    bar = create_plot(feature, portfolio)
    bar = json.loads(bar)

    return render_template('result.html', plot=bar[0], layout=bar[1])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)

