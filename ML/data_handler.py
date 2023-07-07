from yahoo_fin.stock_info import get_data

# Retrieve data
DATA_WEEKLY = get_data(
                        ticker = "amzn", 
                        start_date = "7/07/2003", 
                        end_date = "7/07/2023", 
                        index_as_date = True, 
                        interval = "1wk"
                        )
                        
# Create new column for each day stating tomorrow's closing price for the stock
DATA_WEEKLY["TomorrowClose"] = DATA_WEEKLY["close"].shift(-1)

# Set the target as 1 if the price of the stock increases tomorrow, otherwise set it to 0 (1 = Price went up, 0 = Price stayed the same / went down)
DATA_WEEKLY["Target"] = (DATA_WEEKLY["TomorrowClose"] > DATA_WEEKLY["close"]).astype(int)


periods = [2, 5, 60, 250, 500] # 250 = 1 trading year
for p in periods:

    # Closing price ratios:

    # Find the average closing price of the last p days
    rolling_averages =  DATA_WEEKLY.rolling(window = p).mean()  
    cr_column_name = f"CloseRation_{p}"
    # Find the ratio closing price and the average closing price over the last p days (Inclusive of the current day)
    DATA_WEEKLY[cr_column_name] = DATA_WEEKLY["close"] / rolling_averages["close"]

    # Trend over the past few days
    t_column_name = f"Trend_{p}"
    DATA_WEEKLY[t_column_name] = DATA_WEEKLY.shift(1).rolling(p).sum()["Target"] # Sums up the targets over the last p days (Not including today's target)
    

# Removes rows which contain "NaN" inside of any columns
DATA_WEEKLY = DATA_WEEKLY.dropna()

print(DATA_WEEKLY)
