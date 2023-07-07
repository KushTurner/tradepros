from yahoo_fin.stock_info import get_data
from torch import float32 as torch_float_32
from torch import tensor as torch_tensor

class DataHandler:

    def __init__(self):
        pass
    
    def retrieve_data(self, ticker, start_date, end_date, interval):

        # Retrieve data
        DATA = get_data(
                        ticker = ticker, 
                        start_date = start_date, 
                        end_date = end_date, 
                        index_as_date = True, 
                        interval = interval
                        )
        
        # Remove "ticker" column
        DATA = DATA.drop("ticker", axis = 1)

        # Create new column for each day stating tomorrow's closing price for the stock
        DATA["TomorrowClose"] = DATA["close"].shift(-1)

        # Set the target as 1 if the price of the stock increases tomorrow, otherwise set it to 0 (1 = Price went up, 0 = Price stayed the same / went down)
        DATA["Target"] = (DATA["TomorrowClose"] > DATA["close"]).astype(int)
        
        # Periods based on the interval (daily, weekly, monthly)
        if interval == "1d":
            periods = [2, 5, 60, 250, 500] 
        elif interval == "1wk":
            periods = [1, 3, 9, 36, 108] # 1 week, 1 month, 3 months, 1 year
        else:
            periods = [1, 3, 9, 12, 24]
        
        for p in periods:

            # Closing price ratios:

            # Find the average closing price of the last p days/weeks/months
            rolling_averages =  DATA["close"].rolling(window = p).mean()  

            cr_column_name = f"CloseRation_{p}"
            # Find the ratio closing price and the average closing price over the last p days/weeks/months (Inclusive of the current day)
            DATA[cr_column_name] = DATA["close"] / rolling_averages

            # Trend over the past few days
            t_column_name = f"Trend_{p}"
            DATA[t_column_name] = DATA.shift(1).rolling(p).sum()["Target"] # Sums up the targets over the last p days/weeks/months (Not including today's target)

        # Removes rows which contain "NaN" inside of any columns
        DATA.dropna(inplace = True)

        # Set all columns in the data to the float datatype (All values must be homogenous when passed as a tensor into a model)
        DATA = DATA.astype(float)
        
        # Convert the pandas dataframe into a PyTorch tensor
        return self.dataframe_to_ptt(pandas_dataframe = DATA, desired_dtype = torch_float_32)
    
    def dataframe_to_ptt(self, pandas_dataframe, desired_dtype = torch_float_32):
        
        # Converts a pandas dataframe to a PyTorch tensor (With the default dtype being torch.float32)
        # 1. pandas_dataframe.values extracts the data into numpy arrays (dtype = float64) [Maintains shape]
        # 2. torch_tensor converts np array to PyTorch tensor

        return torch_tensor(pandas_dataframe.values, dtype = desired_dtype)


DH = DataHandler()
data = DH.retrieve_data(
                ticker = "amzn", 
                start_date = "7/07/2003", 
                end_date = "7/07/2023", 
                interval = "1wk" 
                )
print(data.shape)
print(data.isnan().any().item()) # Check if the tensor contains "nan"