from yahoo_fin.stock_info import get_data


def retrieve_data(ticker, start_date, end_date, interval):

    # Retrieve data
    DATA = get_data(
                    ticker = ticker, 
                    start_date = start_date, 
                    end_date = end_date, 
                    index_as_date = True, 
                    interval = interval
                    )
                            
    # Create new column for each day stating tomorrow's closing price for the stock
    DATA["TomorrowClose"] = DATA["close"].shift(-1)

    # Set the target as 1 if the price of the stock increases tomorrow, otherwise set it to 0 (1 = Price went up, 0 = Price stayed the same / went down)
    DATA["Target"] = (DATA["TomorrowClose"] > DATA["close"]).astype(int)


    periods = [2, 5, 60, 250, 500] # 250 = 1 trading year
    for p in periods:

        # Closing price ratios:

        # Find the average closing price of the last p days
        rolling_averages =  DATA.rolling(window = p).mean()  
        cr_column_name = f"CloseRation_{p}"
        # Find the ratio closing price and the average closing price over the last p days (Inclusive of the current day)
        DATA[cr_column_name] = DATA["close"] / rolling_averages["close"]

        # Trend over the past few days
        t_column_name = f"Trend_{p}"
        DATA[t_column_name] = DATA.shift(1).rolling(p).sum()["Target"] # Sums up the targets over the last p days (Not including today's target)
        

    # Removes rows which contain "NaN" inside of any columns
    DATA = DATA.dropna()

    return DATA


D = retrieve_data(
                ticker = "amzn", 
                start_date = "7/07/2003", 
                end_date = "7/07/2023", 
                interval = "1wk"
                )

print(D)