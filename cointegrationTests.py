def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as sm
    import statsmodels.tsa.stattools as ts
    # import statsmodels.tsa.vector_ar.vecm as vm

    listOfETFs = ['XLV', 'PTH']
    storage_location = "C:\\Users\\hsuen\\Desktop\\QuantTrading\\ETFPair\\data\\"
    #############################################################################

    meta_data = pd.DataFrame(columns=listOfETFs)
    # read in the data
    for ETF in listOfETFs:
        x = pd.read_csv(storage_location + '{0}.csv'.format(ETF))
        # read in only the last 500 time steps(since we know those dates will match up
        last_indices = x.tail(500).index
        x = x.iloc[last_indices]
        meta_data[ETF] = x['close']

    # scatter plot of the two ETFs
    meta_data.plot.scatter(x=listOfETFs[0], y=listOfETFs[1])
    #plt.show()

    # switch the two plots
    meta_data.plot.scatter(x=listOfETFs[1], y=listOfETFs[0])

    # perform linear regression to determine a hedge ratio
    results = sm.ols(formula='XLV ~ PTH', data=meta_data[['PTH', 'XLV']]).fit()
    print(results.params)

    # run the co-integrated ADF test to see what if the co-integrated stock is
    # mean-reverting
    coint_t, pvalue, crit_value = ts.coint(meta_data['XLV'], meta_data['PTH'])
    print("The T-statistic: {0}".format(coint_t))
    print("The Critical Value: {0}".format(crit_value))
    print('Cannot Reject Null-Hypothesis that lambda is equal to zero')

    # the Null-Hypothesis that lambda is equal to zero cannot be completely rejected
    # But we can still assume that they are relatively co-integrating and use the value to determine
    # the look-back

    # re-perform regression results, but this time as a measure of the change
    # results.params is a pandas series
    # assume that the XLV is a function of PTH
    hedge_ratio = results.params['PTH']
    # the "values" method of a dataframe returns the dataframe as numpy array
    # but without the column labels
    # we perform matrix multiplication using np.dot (np.dot is NOT element wise multiplication)
    allocation = np.array([1, hedge_ratio])
    portfolio_val = np.dot(meta_data.values, allocation)
    portfolio_lag = pd.DataFrame(portfolio_val).shift()
    delta_port = portfolio_val - portfolio_lag[0]
    df = pd.concat([delta_port,portfolio_lag], axis=1)
    df.columns = ['delta', 'lag']
    # run regression on the delta as a function of itself
    regress_results = sm.ols(formula='delta ~ lag', data=df[['delta', 'lag']]).fit()
    print(regress_results.params)
    print('############')
    half_life = -np.log(2)/regress_results.params['lag']
    print('The half life value is {0}'.format(half_life))

    # test the strategy
    lookback = round(half_life).astype(int)
    hedge_ratios = np.ones((len(meta_data), 1))

    # lets calculate a dynamically changing hedge ratio based on lookback
    for x in range(lookback, len(meta_data)):
        sub_data = meta_data.iloc[x-lookback:x]
        regress_results = sm.ols(formula='XLV ~ PTH', data=sub_data[['XLV', 'PTH']]).fit()
        hedge_ratios[x] = regress_results.params['PTH']

    # perform element wise multiplication to get the dynamic portfolio
    hedge_ratios = np.concatenate((np.ones((len(hedge_ratios), 1)), -1*hedge_ratios), axis=1)

    # hedge_ratios contain the hedge_ratios for all time steps computed based on the lookback period
    # we only have values starting from hedge_ratios[lookback], everything else is just one At the lookback is
    # the first time period where we have enough data to compute the hedge ratio

    # perform element wise multiplication between the actual data values of the two ETFs and their
    # corresponding hedge ratios (1 for XLV and some value for PTH)
    # dynamic_portfolio_val_sum is the TOTAL market value for both ETFs
    dynamic_portfolio_val = np.multiply(meta_data.iloc[lookback:].values, hedge_ratios[lookback:])
    dynamic_portfolio_val_sum = pd.DataFrame(dynamic_portfolio_val).sum(axis=1)

    # num_units tells us the proportion of dollars we should invest given how far away the
    # combination of our portfolio is away from its moving mean
    # However, one caveat. The "rolling" feature in pandas dataframes allows you to
    # calculate the rolling mean, but includes that rightmost datapoint in that mean
    # i.e. the moving mean of 20 points includes the 20th data point in that calculation and
    # puts the mean at the 20th data point
    num_units = -(dynamic_portfolio_val_sum - dynamic_portfolio_val_sum.rolling(lookback).mean())/dynamic_portfolio_val_sum.rolling(lookback).std()

    # num_units_split is just reshaping it so that we can treat it like an array with numpy
    num_units_split = num_units.values.reshape(len(num_units), 1)

    # positions now represents a Tx2 array where we multiply the market value of our
    # positions by the actual dollars allocated to them (num_units). Keep in mind
    # that dynamic_portfolio_val already includes the hedge ratio
    positions = np.multiply(num_units_split, dynamic_portfolio_val)
    position_lag = pd.DataFrame(positions).shift()

    # The lag shifts it back, accounting for the weird behavior we see with the "rolling"
    # situation above with pandas series so there is no look ahead bias

    # now multiply that lagged position by the actual percentage changes to see how our market values
    # changed if we had the position the previous day
    pnl = np.multiply(position_lag, meta_data.iloc[lookback:].pct_change())
    # PNL is giving us how much ACTUAL money we make or lose each day. position_lag is the equity, total market value we
    # have on the previous day. The pct_change (second term) is the percentage change of the stock, which is also the
    # percentage change of our total portfolio. Thus, we multiply the percentage change by our total market value to get
    # the actual dollar amount our portfolio increased or decreased that day
    
    # sum across both ETFs to get total money we make or lose each day
    pnl = pnl.sum(axis=1)

    # now every entry in pnl says how much we made or gained each day. The returns would then be the money we gained
    # divided by the market value we had before
    ret = pnl/np.sum(np.abs(position_lag), axis=1)
    (np.cumprod(1+ret)-1).plot()
    plt.show()

    print("The APR is {0} and the Sharpe is {1}".format((np.prod(1+ret))**(252/len(ret)),
                                                        np.sqrt(252)*np.mean(ret)/np.std(ret)))


    ##### Numpy notes:
    ##### np.dot, which is dot product
    ##### np.multiply which should be element wise, or equivalent to just the * sign
    ##### the last one would then be broadcasting: basically if a dimension doesnt match,
    ##### it will be expanded so that the two arrays are equivalent


if __name__ == '__main__':
    main()
