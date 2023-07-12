from yahoo_fin.stock_info import get_data
from torch import float32 as torch_float_32
from torch import int64 as torch_int_64
from torch import tensor as torch_tensor
from torch import ones as torch_ones
from torch import multinomial as torch_multinomial
from torch import stack as torch_stack

class DataHandler:

    def __init__(self, device, generator):

        self.device = device # CUDA or CPU
        self.generator = generator # Reproducibility

        # self.data - Holds the inputs for each example
        # self.labels - Holds the corresponding targets for each example
        # self.TRAIN_S - Training set
        # self.VAL_S - Validation set
        # self.TEST_S - Test set
        
        # self.n_features - Number of inputs that will be passed into a model (i.e. the number of columns/features in the pandas dataframe)
        
    def retrieve_data(self, ticker, start_date, end_date, interval, normalise = False, standardise = False):

        # Retrieve data
        DATA = get_data(
                        ticker = ticker, 
                        start_date = start_date, 
                        end_date = end_date, 
                        index_as_date = True, 
                        interval = interval
                        )
        
        # Remove "ticker" column
        DATA.drop("ticker", axis = 1, inplace = True)

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

        # Shuffle data (Do this before separating the labels from the dataframe)
        ##########################################

        # Separate the labels from the main dataframe (the other columns will be used as inputs)
        labels = DATA["Target"]
        self.labels = self.dataframe_to_ptt(pandas_dataframe = labels, desired_dtype = torch_int_64)
        DATA.drop("Target", axis = 1, inplace = True)
        
        # Normalise or standardise data (If both are True, only normalise the data)
        # Note: Only one should be performed
        if normalise == True:
            # Normalise specific columns (input values to be between 0 and 1)
            # Note: Preserves relative relationships between data points but eliminates differences in magnitude
            cols_to_alter = ["open", "high", "low", "close", "adjclose", "volume", "TomorrowClose"]
            DATA[cols_to_alter] = self.normalise_columns(dataframe = DATA, cols_to_norm = cols_to_alter)
            
        elif standardise == True:
            # Standardise specific columns (mean 0, unit variance)
            # Note: Brings data features onto a similar scale to be comparable (Helps remove the influence of the mean and scale of data where distribution of data is not Gaussian or contains outliers)
            cols_to_alter = ["open", "high", "low", "close", "adjclose", "volume", "TomorrowClose"]
            DATA[cols_to_alter] = self.standardise_columns(dataframe = DATA, cols_to_standard = cols_to_alter)
        
        # Convert the pandas dataframe into a PyTorch tensor and save the data as an attribute
        self.data = self.dataframe_to_ptt(pandas_dataframe = DATA, desired_dtype = torch_float_32)
        
        # Set the number of features that will go into the first layer of a model
        self.n_features = self.data.shape[1]
    
    def normalise_columns(self, dataframe, cols_to_norm):
        # Return normalised columns
        return (dataframe[cols_to_norm] - dataframe[cols_to_norm].min()) / (dataframe[cols_to_norm].max() - dataframe[cols_to_norm].min())
    
    def standardise_columns(self, dataframe, cols_to_standard):
        # Mean of all columns
        mean = dataframe[cols_to_standard].mean()
        # Std of all columns
        std = dataframe[cols_to_standard].std()
        
        # Return standardised columns
        return (dataframe[cols_to_standard] - mean) / std

    def dataframe_to_ptt(self, pandas_dataframe, desired_dtype = torch_float_32):
        
        # Converts a pandas dataframe to a PyTorch tensor (With the default dtype being torch.float32)
        # 1. pandas_dataframe.values extracts the data into numpy arrays (dtype = float64) [Maintains shape]
        # 2. torch_tensor converts np array to PyTorch tensor

        return torch_tensor(pandas_dataframe.values, dtype = desired_dtype)

    def generate_batch(self, batch_size, split_selected):
        
        # Find the inputs and labels in the split selected and find the number of examples in this split
        inputs, labels = getattr(self, f"{split_selected.upper()}_S") # i.e. self.TRAIN_S, self.VAL_S, self.TEST_S
        num_examples = labels.shape[0]
        
        # Generate indexes which correspond to each example in the labels and inputs of this split (perform using CUDA if possible)
        u_distrib = torch_ones(num_examples, device = self.device) / num_examples # Uniform distribution
        example_idxs = torch_multinomial(input = u_distrib, num_samples = batch_size, replacement = True, generator = self.generator)

        # Move indices to the "cpu" so that we can index the dataset, which is currently stored on the CPU
        example_idxs = example_idxs.to(device = "cpu")

        # Return the examples and the corresponding targets (for predicting whether the price goes up or down for the next day)
        # Note: If self.device == "cuda", then the batch will be moved back onto the GPU
        return inputs[example_idxs].to(device = self.device), labels[example_idxs].to(device = self.device) 

    def create_splits(self, num_context_days = None):
        
        # Split distribution percentages (Will be modified depending on the num_context_days)
        split_idx = {
                    "Train": 0.8,
                    "Val": 0.1,
                    "Test": 0.1
                    }
        
        # MLP 
        if num_context_days == None:
            
            # data.shape = [number of single day examples, number of features for each day]
            # labels.shape = [number of single day examples, correct prediction for stock trend for the following day]

            # Update split indexes
            total_examples = self.data.shape[0]
            for split_name in split_idx.keys():
                split_idx[split_name] = int(split_idx[split_name] * total_examples)
            
            # Cut off between train and val split
            val_end_idx = split_idx["Train"] + split_idx["Val"]

        # RNN
        else:
            # Let num_context_days = 10, batch_size = 32
            # Single batch should be [10 x [32 * num_features] ]
            # 32 x [ClosingP, OpeningP, Volume, etc..] 10 days ago
            # The next batch for the recurrence will be the day after that day
            # 32 x [ClosingP, OpeningP, Volume, etc..] 9 days ago
            # Repeats until all 10 days have been passed in (for a single batch)

            # Find the amount of total sequences we can have (4510 examples = 451 sequences) 
            # Note: Cut off from the start of the data and labels as they will be older data
            remainder_days = self.data.shape[0] % num_context_days
            self.data = self.data[remainder_days:]
            self.labels = self.labels[remainder_days:]

            # Convert the data and labels into sequences of 10 consecutive days
            # data.shape = (number of 10 consecutive days sequences, 10 consecutive days of examples, number of features in each day)
            # labels.shape = (number of 10 consecutive days sequences, 10 correct predictions for the stock trend for the following day)
            self.data = torch_stack([self.data[i:i + num_context_days] for i in range(0, self.data.shape[0], num_context_days)], dim = 0)
            self.labels = torch_stack([self.labels[i:i + num_context_days] for i in range(0, self.labels.shape[0], num_context_days)], dim = 0)

            # Update split indexes
            total_examples = self.data.shape[0]
            for split_name in split_idx.keys():
                split_idx[split_name] = int(split_idx[split_name] * total_examples)

            # Cut off between train and val split
            val_end_idx = split_idx["Train"] + split_idx["Val"]
            
        # Create the splits, each tuple = (inputs, labels)
        self.TRAIN_S = (self.data[0:split_idx["Train"]], self.labels[0:split_idx["Train"]])
        self.VAL_S = (self.data[split_idx["Train"]:val_end_idx], self.labels[split_idx["Train"]:val_end_idx])
        self.TEST_S = (self.data[val_end_idx:], self.labels[val_end_idx:])

        # Clear memory
        del self.data
        del self.labels