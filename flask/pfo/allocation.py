import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader as web
from dateutil.relativedelta import relativedelta
import sys
from scipy import optimize

class Asset():
  def __init__(self, ticker, name, start, end):
    self.ticker = ticker
    self.name = name
    
    self.start = start
    self.end = end

    # 데이터 불러오기
    self.data = web.get_data_yahoo(self.ticker, self.start, self.end)
    self.date = self.data.index

    self.price = self.data['Adj Close']
    self.ror = self.price.pct_change().to_list()

    self.asset_df = pd.DataFrame(columns=['Price', 'RoR'])
    self.asset_df['Price'] = self.price
    self.asset_df['RoR'] = self.ror

  def get_date(self):
    return self.date

  def get_ror(self):
    return self.ror

  def get_name(self):
    return self.name

  def get_df(self):
    return self.asset_df


class GMV_Asset(Asset):
    def __init__(self, ticker_list, start, end):
        self.ticker_list = ticker_list
        self.start = start
        self.end = end
        self.df = pd.DataFrame(columns=self.ticker_list)
        self.asset_list = []
        
    def create_df(self):
        for stock in self.ticker_list:
            asset = Asset(stock, stock, self.start, self.end)
            self.asset_list.append(asset)
            self.df[stock] = asset.data["Adj Close"]

            if stock == self.ticker_list[0]:
                self.df['date'] = asset.data.index
                self.df.set_index('date', inplace=True)
            
    def get_returns(self):
        return self.df.pct_change()

    def get_mean_returns(self):
        return self.get_returns().mean()
    
    def get_cov_matrix(self):
        return self.get_returns().cov()


class AWF_Asset(Asset):
    def __init__(self, equity_ticker_list, inl_ticker_list, corp_ticker_list,
                 bond_ticker_list, emd_ticker_list, comm_ticker_list, start, end):
        self.start = start
        self.end = end
        self.result = []
        self.asset_list = []
        self.total_ticker_dict = {
            "equity_ticker_list" : equity_ticker_list,
            "inl_ticker_list" : inl_ticker_list,
            "corp_ticker_list" : corp_ticker_list,
            "bond_ticker_list" : bond_ticker_list,
            "emd_ticker_list" : emd_ticker_list,
            "comm_ticker_list" : comm_ticker_list
        }
        self.result_dict = {
            "equity" : [],
            "inl" : [],
            "corp" : [],
            "bond" : [],
            "emd" : [],
            "comm" : []
        }
        
    def create_dict(self):
        for t_key in self.total_ticker_dict:
            for stock in self.total_ticker_dict[t_key]:
                asset = Asset(stock, stock, self.start, self.end)
                self.asset_list.append(asset)
                value = asset.data["Adj Close"].pct_change().dropna()
                
                key = t_key.split("_")[0]
                self.result_dict[key].append(value)
                
    def calculate_value(self):
        for label in self.result_dict:
            value = self.result_dict[label][0]
            
            for i in range(len(self.result_dict[label]) - 1):
                value += self.result_dict[label][i]
            
            avg = value / len(self.result_dict[label])
            self.result.append(avg)
        return self.result
    
    def get_value(self):
        self.create_dict()
        return self.calculate_value(), self.total_ticker_dict


class Portfolio():
    def __init__(self, name, assets, ratios, initial_balance, rebalancing_interval, look_back, category):
        self.name = name
        self.assets = assets
        self.category = category 

        dates = [each.get_date() for each in self.assets]
        lendates = [len(each) for each in dates]
        print(lendates, file=sys.stderr)
        assert len(set(lendates)) == 1
        self.date = dates[0]
        self.look_back = look_back
        self.look_back_date = self.date[0] - relativedelta(months=self.look_back)

        self.initial_balance = initial_balance
        self.rebalancing_interval = rebalancing_interval
        self.ratios = [each / sum(ratios) for each in ratios]

        self.backtest_df = self.backtest()
        self.backtest_result_df = self.backtest_result()
        self.annual_df = self.periodic_result('annual')
        self.monthly_df = self.periodic_result('monthly')
        self.summary = self.summarize()
            
    def backtest(self):
        balances = []

        if self.name == "GMV":
            gmv_asset = GMV_Asset([tick for cat in self.category for tick in cat], self.look_back_date, self.date[0])
            gmv_asset.create_df()
            gmv = GMVPortfolio(gmv_asset.get_mean_returns(), gmv_asset.get_cov_matrix(), 25000, 0.0178)
            self.ratios = [round(i, 2)for i in gmv.min_variance().x]

        elif self.name == "All Weather":
            awf_asset = AWF_Asset(self.category[0], self.category[1], self.category[2], 
                                    self.category[3], self.category[4], self.category[5], self.look_back_date, self.date[0])
            awf_ret, awf_ticker = awf_asset.get_value()
            self.ratios = AllWeatherPortfolio(awf_ret, awf_ticker).awf()

        for i in range(len(self.assets)):
            balance = [self.initial_balance * self.ratios[i]]
            balances.append(balance)

        total_balance = [self.initial_balance]
        next_rebalancing = [self.date[0] + relativedelta(months=self.rebalancing_interval)]

        for i in range(1, len(self.date)):
            total_balance_tmp = 0

            if self.date[i] >= next_rebalancing[i-1]: # 리밸런싱하는 날
                self.look_back_date = self.date[i] - relativedelta(months=self.look_back)
                next_rebalancing.append(next_rebalancing[i-1] + relativedelta(months=self.rebalancing_interval))

                if self.name == "GMV":
                    gmv_asset = GMV_Asset([tick for cat in self.category for tick in cat], self.look_back_date, self.date[i])
                    gmv_asset.create_df()
                    gmv = GMVPortfolio(gmv_asset.get_mean_returns(), gmv_asset.get_cov_matrix(), 25000, 0.0178)
                    self.ratios = [round(i, 2)for i in gmv.min_variance().x]

                elif self.name == "All Weather":
                    awf_asset = AWF_Asset(self.category[0], self.category[1], self.category[2], 
                                          self.category[3], self.category[4], self.category[5], self.look_back_date, self.date[i])
                    awf_ret, awf_ticker = awf_asset.get_value()
                    self.ratios = AllWeatherPortfolio(awf_ret, awf_ticker).awf()
                    
                for j in range(len(self.assets)):
                    balance = total_balance[i-1] * self.ratios[j] * (1 + self.assets[j].get_ror()[i])
                    balances[j].append(balance)
                    total_balance_tmp += balances[j][i]
            else:
                next_rebalancing.append(next_rebalancing[i-1])
                for j in range(len(self.assets)):
                    balances[j].append(balances[j][i-1] * (1 + self.assets[j].get_ror()[i]))
                    total_balance_tmp += balances[j][i]

            total_balance.append(total_balance_tmp)
        
        df = pd.DataFrame()
        df['Date'] = self.date
        df.set_index('Date', inplace=True)
        df['Total'] = total_balance

        for i in range(len(self.assets)):
            df[self.assets[i].get_name()] = balances[i]

        self.backtest_df = df

        return df 

    def backtest_result(self):
        df = pd.DataFrame()
        df['Date'] = self.date
        df.set_index('Date', inplace=True)
        label = ['Rate of Return', 'Cumulative Return', 'CAGR', 'Drawdown', 'MDD']

        result = dict()
        result["Total"] = self.balance_result(self.backtest_df["Total"].to_list())
        df['Total Balance'] = self.backtest_df["Total"].to_list()
      
        for i in range(len(label)):
            df[f'Total {label[i]}'] = result["Total"][i]

        self.backtest_result_df = df

        return df

    def balance_result(self, balance):
        ror, cumr = [0], [0]
        cagr, stdev, sharpe = [0], [0], [0]

        time_period = (self.date[-1] - self.date[0]).days / 365

        for i in range(1, len(self.date)):
            ror.append((balance[i] / balance[i-1] - 1) * 100)
            cumr.append((balance[i] / balance[0] - 1) * 100)
            cagr.append(((balance[i] / balance[0]) ** (1 / float(time_period)) - 1) * 100)

        time_period = (self.date[-1] - self.date[0]).days / 365

        max_balance = [balance[0]]
        for i in range(1, len(balance)):
            max_balance.append(max(balance[0:i+1]))
        assert len(balance) == len(max_balance)

        drawdown, mdd = [], []
        for i in range(len(balance)):
            drawdown.append((balance[i] - max_balance[i]) / max_balance[i])
            mdd.append(min(drawdown))

        return ror, cumr, cagr, drawdown, mdd, stdev, sharpe

    def periodic_result(self, mode):
        df = pd.DataFrame()

        label = 'Total'
        return_points, returns = [], []
        start_balance, end_balance = [], []
        start = self.backtest_df[label].to_list()[0]

        if mode == 'annual':
            for i in range(1, len(self.date)):
                if self.date[i].year != self.date[i-1].year:
                    return_points.append(self.date[i-1].year)
                    returns.append((self.backtest_df[label].to_list()[i-1] / start - 1) * 100)
                    start_balance.append(start)
                    end_balance.append(self.backtest_df[label].to_list()[i-1])
                    start = self.backtest_df[label].to_list()[i]
                elif self.date[i] == self.date[-1]: # 마지막 거래일
                    return_points.append(self.date[i].year)
                    returns.append((self.backtest_df[label].to_list()[i] / start - 1) * 100)
                    start_balance.append(start)
                    end_balance.append(self.backtest_df[label].to_list()[i-1])
            df[f'{label} {mode.capitalize()} Return'] = returns
        
        elif mode == 'monthly':
            for i in range(1, len(self.date)):
                if self.date[i].month != self.date[i-1].month:
                    return_points.append(self.date[i-1].strftime('%Y-%m'))
                    returns.append((self.backtest_df[label].to_list()[i-1] / start - 1) * 100)
                    start_balance.append(start)
                    end_balance.append(self.backtest_df[label].to_list()[i-1])
                    start = self.backtest_df[label].to_list()[i]
                elif self.date[i] == self.date[-1]: # 마지막 거래일
                    return_points.append(self.date[i].strftime('%Y-%m'))
                    returns.append((self.backtest_df[label].to_list()[i] / start - 1) * 100)
                    start_balance.append(start)
                    end_balance.append(self.backtest_df[label].to_list()[i-1])
            df[f'{label} {mode.capitalize()} Return'] = returns

        df[f'Return {mode.capitalize()}'] = return_points
        df.set_index(f'Return {mode.capitalize()}', inplace=True)

        return df

    def summarize(self):
        detail = ''
        for i in range(len(self.assets)):
            name = self.assets[i].get_name()
            percentage = int(self.ratios[i] * 100)

            detail += f'{name} ({percentage}%) ' 

        monthly = self.monthly_df['Total Monthly Return']   
        sharpe = np.mean(monthly) / np.std(monthly)

        return [detail, self.backtest_result_df['Total Balance'][0], self.backtest_result_df['Total Balance'][-1],
                str(round(self.backtest_result_df['Total CAGR'][-1], 2))+'%', 
                str(round(self.backtest_result_df['Total MDD'][-1] * 100, 2))+'%',
                round(sharpe, 2)]

    def get_name(self):
        return self.name

    def get_date(self):
        return self.date

    def get_backtest(self):
        return self.backtest_df

    def get_backtest_result(self):
        return self.backtest_result_df

    def get_annual_result(self):
        return self.annual_df

    def get_monthly_result(self):
        return self.monthly_df

    def get_summary(self):    
        return self.summary


#class portfolio takes matrix of returns for every stock, and array of their weights
class GMVPortfolio:
    def __init__(self, mean_returns, cov_matrix, num_portfolios, risk_free_rate):
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.num_portfolios = num_portfolios
        self.risk_free_rate = risk_free_rate

    def neg_sharpe_ratio(self, weights, mean_returns, cov_matrix, risk_free_rate):
        p_var, p_ret = self.portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_var

    def max_sharpe_ratio(self):
        num_assets = len(self.mean_returns)
        args = (self.mean_returns, self.cov_matrix, self.risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0,1.0)
        bounds = tuple(bound for asset in range(num_assets))
        result = optimize.minimize(self.neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    
    def portfolio_volatility(self, weights, mean_returns, cov_matrix):
        return self.portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

    def min_variance(self):
        num_assets = len(self.mean_returns)
        args = (self.mean_returns, self.cov_matrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0.0,1.0)
        bounds = tuple(bound for asset in range(num_assets))

        result = optimize.minimize(self.portfolio_volatility, num_assets*[1./num_assets,], args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)

        return result

    def portfolio_return(self, weights):
        return self.portfolio_annualised_performance(weights, self.mean_returns, self.cov_matrix)[1]
    
    def efficient_return(self, target):
        num_assets = len(self.mean_returns)
        args = (self.mean_returns, self.cov_matrix)

        constraints = ({'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - target},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0,1) for asset in range(num_assets))
        result = optimize.minimize(self.portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    def efficient_frontier(self, returns_range):
        efficients = []
        for ret in returns_range:
            efficients.append(self.efficient_return(ret))
        return efficients
    
    def portfolio_annualised_performance(self, weights, mean_returns, cov_matrix):
        returns = np.sum(self.mean_returns*weights ) *252
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
        return std, returns
    
    def random_portfolios(self):
        results = np.zeros((3, self.num_portfolios))
        weights_record = []
        for i in range(self.num_portfolios):
            weights = np.random.random(self.mean_returns.shape[0])
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std_dev, portfolio_return = self.portfolio_annualised_performance(weights, self.mean_returns, self.cov_matrix)
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            results[2,i] = (portfolio_return - self.risk_free_rate) / portfolio_std_dev
        return results, weights_record       
        
    def display_ef_with_selected_and_random(self, returns, cols):
        results, _ = self.random_portfolios()
        
        max_sharpe = self.max_sharpe_ratio()
        sdp, rp = self.portfolio_annualised_performance(max_sharpe['x'], self.mean_returns, self.cov_matrix)
        max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=cols,columns=['allocation'])
        max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
        max_sharpe_allocation = max_sharpe_allocation.T

        min_vol = self.min_variance()
        sdp_min, rp_min = self.portfolio_annualised_performance(min_vol['x'], self.mean_returns, self.cov_matrix)
        min_vol_allocation = pd.DataFrame(min_vol.x,index=cols,columns=['allocation'])
        min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
        min_vol_allocation = min_vol_allocation.T

        an_vol = np.std(returns) * np.sqrt(252)
        an_rt = self.mean_returns * 252

        print("-"*80)
        print("Maximum Sharpe Ratio Portfolio Allocation\n")
        print("Annualised Return:", round(rp,2))
        print("Annualised Volatility:", round(sdp,2))
        print("\n")
        print(max_sharpe_allocation)
        print("-"*80)
        print("Minimum Volatility Portfolio Allocation\n")
        print("Annualised Return:", round(rp_min,2))
        print("Annualised Volatility:", round(sdp_min,2))
        print("\n")
        print(min_vol_allocation)
        print("-"*80)
        print("Individual Stock Returns and Volatility\n")
        for i, txt in enumerate(cols):
            print(txt,":","annuaised return",round(an_rt[i],2),", annualised volatility:",round(an_vol[i],2))
        print("-"*80)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(an_vol,an_rt,marker='o',s=200)
        ax.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)

        for i, txt in enumerate(cols):
            ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
        ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
        ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')

        target = np.linspace(rp_min, 0.34, 50)
        efficient_portfolios = self.efficient_frontier(target)
        ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
        ax.set_title('Portfolio Optimization with Individual Stocks')
        ax.set_xlabel('annualised volatility')
        ax.set_ylabel('annualised returns')
        ax.legend(labelspacing=0.8)        


class AllWeatherPortfolio(Asset):
    
    def __init__(self, dictionary, ticker):
        self.result = dictionary
        self.total_ticker_dict = ticker
   
    def merge(self):
        port = pd.concat(self.result, axis =1 )
        port.columns = ['equity', 'infl', 'corp', 'bond', 'emd', 'comm']
        asset_num = len(port.columns)

        port_gr_r = (self.result[0] + self.result[2] + self.result[4] + self.result[5])/4
        port_gr_f = (self.result[1] + self.result[4] + self.result[5])/3
        port_inf_r = (self.result[1] + self.result[3])/2
        port_inf_f = (self.result[0] + self.result[3])/2

        portfolio = pd.concat([port_gr_r, port_gr_f, port_inf_r, port_inf_f], axis =1 )
        portfolio.columns = ['gr_r', 'gr_f', 'inf_r', 'inf_f' ]
        self.covmat = portfolio.cov()

    def RiskParity_objective(self, x) :
        variance = x.T @ self.covmat @ x
        sigma = variance ** 0.5
        mrc = 1/sigma * (self.covmat @ x)
        rc = x * mrc  
        rc = rc/rc.sum() 
        a = pd.DataFrame(index = range(len(x)), columns = range(len(x)))
        for i in range (0, len(x)):
            for j in range(0, len(x)):
              a[i][j]  = rc[i] - rc[j]
        sum_risk_diffs_squared = np.sum(np.square(np.ravel(a)))
        return sum_risk_diffs_squared


    def weight_sum_constraint(self, x):
        result = x.sum() - 1.0
        return result

    def weight_longonly(self, x):
        return x

    def RiskParity(self):
        x0 = np.repeat(1 / self.covmat.shape[1], self.covmat.shape[1])
        con1 = {'type' : 'eq', 'fun' : self.weight_sum_constraint}
        con2 = {'type': 'ineq', 'fun' : self.weight_longonly}  
        constraint =  ([con1, con2])
        options = {'ftol' : 1e-20, }
        result = optimize.minimize(fun = self.RiskParity_objective,
                    x0 = x0,
                    method = 'SLSQP',
                    constraints = constraint,
                    options = options)
        return result.x


    def weight(self, wt_erc):
        weight = [] 
        
        for label in self.total_ticker_dict:
            if label == "equity_ticker_list":
                length = len(self.total_ticker_dict[label])
                for i in range(length):
                    weight.append(wt_erc[0]/(4*length) + wt_erc[3]/(2*length))              
            elif label == "inl_ticker_list":
                weight.append(wt_erc[1]/3 +  wt_erc[2]/2)            
            elif label == "corp_ticker_list":
                weight.append(wt_erc[0]/4)
            elif label == "bond_ticker_list":
                length = len(self.total_ticker_dict[label])
                for i in range(length):
                    weight.append(wt_erc[2]/(2*length) + wt_erc[3]/(2*length))               
            elif label == "emd_ticker_list":
                length = len(self.total_ticker_dict[label])
                for i in range(length):
                    weight.append(wt_erc[0]/(4*length) + wt_erc[1]/(3*length))
            elif label == "comm_ticker_list":
                for i in range(length):
                    weight.append(wt_erc[0]/(4*length) + wt_erc[1]/(3*length))

        return weight

    def awf(self):
        self.merge()
        wt_erc = self.RiskParity()
        x = self.weight(wt_erc)

        return x


