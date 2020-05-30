def main():
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    import numpy as matrix
    storage_location = "C:\\Users\\hsuen\\Desktop\\QuantTrading\\alpaca\\data\\StockData\\"

    # get the constituent stocks in the S&P500
    spList = pd.read_csv('data/sp500.csv')

    # sort the stocks based on their historic "low" values
    sortedSpList = spList.sort_values(by=['Low'])

    meta_data = np.zeros((500, 200))

    # let us extract the stock data from all the csv files
    num_stocks_portfolio = 0
    for x in range(5):
        currRow = sortedSpList.iloc[x]
        current_symbol = currRow['Symbol']
        # only take stocks that have values for the previous 500 trading days
        current_stock = pd.read_csv(storage_location + "{0}.csv".format(current_symbol))
        if len(current_stock) <= 500:
            continue

        current_stock_close = current_stock['close']
        meta_data[:, num_stocks_portfolio] = current_stock_close[-500:]
        num_stocks_portfolio += 1

    # eliminate last n columns that were not used

    # IF YOU GET AN ERROR about divide by zero. There are repeats on 4/24
    # sometimes need to delete those repeats
    meta_data = meta_data[:-1, :]
    meta_data = meta_data[:, :num_stocks_portfolio]
    meta_data_lag = meta_data[:-1, :]
    meta_data = meta_data[1:, :]

    # compute daily returns
    daily_returns = np.divide(np.subtract(meta_data, meta_data_lag), meta_data_lag)

    # compute market returns
    market_returns = np.mean(daily_returns, axis=1)

    market_returns = np.tile(market_returns, (num_stocks_portfolio, 1))
    market_returns = np.matrix.transpose(market_returns)

    weights = -(daily_returns - market_returns)
    weights = weights/pd.DataFrame(np.abs(weights)).sum(axis=1).values.reshape((weights.shape[0], 1))
    # the part after the division creates a 1D matrix that we then divide each of the columns of
    # weights by

    # the normalization of the weights means that we have the SAME amount of capital
    # in this case 1 dollar, constantly being re-divided

    totalDailyRet = (pd.DataFrame(weights).shift()*daily_returns).sum(axis=1)
    ((1+totalDailyRet).cumprod()-1).plot()
    # The subtract by 1 is to get everything back in terms of returns
    # you need to add the one in the beginning so the numbers
    # get BIGGER, but the one does not add anything to the overall
    # returns since it is just being "one" each time. Subtract it out at the
    # end to get back just a decimal value representing the return on your investment
    plt.pyplot.show()

    # we want to see how much this portfolio will net us EACH year
    # thus, we raise it to the fraction of years that it is.
    # i.e., if the length of the totalDaily return is 6 years,
    # you want to raise your total return by 1/6. Thus, 252/(6*252)
    # subtract by 1 to convert back to decimal, since the 1 doesn't do anything besides
    # keep being multiplied by itslef in an exponent to be 1.
    APR = (np.prod(1+totalDailyRet))**(252/len(totalDailyRet)) -1
    # double multiplication sign here mean take exponent
    Sharpe = np.sqrt(252)*np.mean(totalDailyRet)/np.std(totalDailyRet)

    #NOTE: cumprod will create an increasing vector of cumulative product at each point,
    # whereas the product function just calculates the last element (all of the pro
    print("This is APR")
    print(APR)
    print("This is Sharpe")
    print(Sharpe)


if __name__ == '__main__':
    main()
