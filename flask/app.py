import sys
import json
import plotly
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pfo.allocation import Asset, Portfolio
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

final_pfo = dict()
pfo_num = -1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post', methods=['POST'])
def post():
    # if request.method == "POST":
    data = json.loads(request.data.decode('UTF-8'))
    
    start, end = data["start"], data["end"]
    initial, rebalancing = int(data["initial"]), int(data["rebalancing"])

    portfolios = []
    pfo_num = len(data["profiles"])

    for i in range(len(data["profiles"])):
        pfo = data["profiles"][i]
        pfo_name = pfo["profile_name"]

        assets, tickers, asset_names, weights, leverages = [], [], [], [], []

        for j in range(len(pfo["assets"])):
            tickers.append(pfo["assets"][j]["ticker"])
            asset_names.append(pfo["assets"][j]["asset_name"])
            weights.append(pfo["assets"][j]["weight"])
            leverages.append(pfo["assets"][j]["leverage"])
            
        tickers = [t for t in tickers if t]
        asset_names = [n for n in asset_names if n]
        weights = [float(w) for w in weights if w]
        leverages = [int(l) for l in leverages if l]

        lens = [len(tickers), len(weights), len(leverages)]
        assert len(set(lens)) == 1
        
        for k in range(len(tickers)):
            asset = Asset(tickers[k], asset_names[k], start, end)
            assets.append(asset)
        
        portfolios.append(Portfolio(pfo_name, assets, weights, leverages, initial, rebalancing))
        final_pfo[i] = Portfolio(pfo_name, assets, weights, leverages, initial, rebalancing)
        
    print(1, final_pfo, file=sys.stderr)

        # return render_template('result.html', tables=[backtest_df.to_html(classes='data')], titles=backtest_df.columns.values)


    # start = request.form.get("start_date")
    # end = request.form.get("end_date")
    # print(start, end, file=sys.stderr)
    # exit()
    # tickers = [t for t in request.form.getlist("asset[]") if t]
    # names = [n for n in request.form.getlist("name[]") if n]
    # weights = [int(w) for w in request.form.getlist("weight[]") if w]
    # leverages = [int(l) for l in request.form.getlist("leverage[]") if l]
    
    # lens = [len(tickers), len(weights), len(leverages)]
    # assert len(set(lens)) == 1

    # assets = []
    # for i in range(len(tickers)):
    #     asset = Asset(tickers[i], names[i], start, end)
    #     assets.append(asset)
    # print(len(assets), assets[0].get_name(), assets[1].get_name(), file=sys.stderr)

    # return render_template("result.html", assets=assets, weights=weights, leverages=leverages)

    print(portfolios[0], file=sys.stderr)
    # backtest_df = portfolios[0].backtest()
    # print(backtest_df, file=sys.stderr)

    return render_template('result.html')

@app.route('/result')
def result():

    # 여기 걸어봤자 이미 저쪽은 리다이렉트되어있는거라 아무런 도움이 안됨 시벌
    while True:
        if len(final_pfo) == pfo_num:
            print(2, final_pfo, file=sys.stderr)
            return render_template('result.html')

    # return render_template('result.html')
    # return render_template("result.html", tables=[backtest_df.to_html(classes='data')], titles=backtest_df.columns.values)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)

