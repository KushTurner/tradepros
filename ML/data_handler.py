from yahoo_fin.stock_info import get_data, get_dividends
from torch import float32 as torch_float_32
from torch import int64 as torch_int_64
from torch import tensor as torch_tensor
from torch import ones as torch_ones
from torch import multinomial as torch_multinomial
from torch import stack as torch_stack
from torch import randperm as torch_randperm
from torch import cat as torch_cat
from torch import chunk as torch_chunk
from torch import argsort as torch_argsort
from pandas import to_datetime as pd_to_datetime
from pandas import read_csv as pd_read_csv
from os.path import exists as os_path_exists
from math import ceil as math_ceil
from torch.cuda import memory_summary as torch_cuda_memory_summary
from os import mkdir as os_mkdir
from os.path import join as os_path_join
from torch import arange as torch_arange
from time import time as get_time

import yfinance as yf


class DataHandler:

    def __init__(self, device, generator):

        self.device = device # CUDA or CPU
        self.generator = generator # Reproducibility

        # self.data - Holds the inputs for each example
        # self.labels - Holds the corresponding targets for each example
        # self.TRAIN_S(N/S) - Training set
        # self.TEST_S(N/S) - Test set
        
        # self.n_features - Number of inputs that will be passed into a model (i.e. the number of columns/features in the pandas dataframe)

    def retrieve_dates(self, tickers, start_date, end_date, interval): 
        self.dates = [] # Dates of all the companies (Used specifically for getting the dates for the TDH to only generate sentiments for dates that are in the historical dataset

        for ticker in tickers:
            # Retrieve the companies data and add it to the list of dates
            DATA = get_data(
                            ticker = ticker, 
                            start_date = start_date, 
                            end_date = end_date, 
                            index_as_date = True, 
                            interval = interval
                            )
            # Modify the data (required otherwise the tweets dataset in the TDH may contain NaN values)
            DATA = self.modify_data(D = DATA)
            self.dates.append(DATA.index.tolist())
            del DATA

    def retrieve_data(self, 
                      tickers, 
                      start_date, 
                      end_date, 
                      interval, 
                      dated_sentiments,
                      hyperparameters, 
                      include_date_before_prediction_date = False, 
                      ):
        
        # Note: transform_after = True means that all of the companies' data will be standardised / normalised together, instead of separately
        self.data = []
        self.labels = []
        self.dates = [] # Dates of all the companies (Used to sort the data sequences into chronological order)
        invalid_tickers = []

        # Single sentiment values from dates the model are predicting the stock trend on
        if hyperparameters["uses_single_sentiments"] == True:
            self.single_sentiments = []

        # Transformation based on "N_OR_S" and "transform_data"
        if hyperparameters["transform_after"] == True:
            transformation = self.standardise_data if hyperparameters["N_OR_S"] == "S" else self.normalise_data
        else:
            transformation = self.standardise_columns if hyperparameters["N_OR_S"] == "S" else self.normalise_columns

        # For each company, modify the data 
        for ticker in tickers:

            # Retrieve historical data
            try:
                DATA = get_data(
                                ticker = ticker, 
                                start_date = start_date, 
                                end_date = end_date, 
                                index_as_date = True, 
                                interval = interval
                                )

            # Historical data does not exist, so skip ticker entirely
            except:
                invalid_tickers.append(ticker)
                continue

            # Adding dividends
            if "dividends" not in hyperparameters["features_to_remove"]:
                try:
                    DIVIDENDS = get_dividends(ticker = ticker, start_date = start_date, end_date = end_date, index_as_date = True)
                    # Re-index using the dates in the the historical data
                    DIVIDENDS = DIVIDENDS.reindex(DATA.index) # Fill value is automatically NaN

                    # Use linear interpolation to calculate + fill in the missing rows
                    """
                    Notes: 
                    - Days before the first dividend will always have N/A as its values, so will be removed after DataHandler.modify_data()
                    - Use method = "time" to consider the time intervals between data points when estimating missing values (Good for time-series data)
                    """
                    DIVIDENDS["dividend"] = DIVIDENDS["dividend"].interpolate(method = "time")

                    # Add dividends column to historical dataset
                    DATA["dividends"] = DIVIDENDS["dividend"]

                    # Removes rows which contain "NaN" inside of any columns
                    DATA.dropna(inplace = True)

                # No dividends found, set as 0s
                except:
                    DATA["dividends"] = [0 for _ in range(DATA.shape[0])]
            
            # Modify the data (e.g. adding more columns, removing columns, etc.)
            DATA = self.modify_data(
                                    D = DATA, 
                                    dated_sentiments = dated_sentiments, 
                                    include_date_before_prediction_date = include_date_before_prediction_date, 
                                    hyperparameters = hyperparameters
                                    )
            # print(DATA)
            # Separate the labels from the main dataframe (the other columns will be used as inputs)
            labels = DATA["Target"]
            self.labels.append(self.dataframe_to_ptt(pandas_dataframe = labels, desired_dtype = torch_int_64))
            DATA.drop("Target", axis = 1, inplace = True)

            # Single sentiments (Used to extract the sentiment values from the dates that the model is predicting the stock trend on)
            if hyperparameters["uses_single_sentiments"]:
                self.single_sentiments.append(self.dataframe_to_ptt(pandas_dataframe = DATA["sentiment_tomorrow"], desired_dtype = torch_float_32))
                DATA.drop("sentiment_tomorrow", axis = 1, inplace = True) # Remove this column as it should not be an input feature.

            # Create normalised and standardised versions of the data
            if hyperparameters["transform_after"] == False: # Standardising / Normalising companies separately
                """
                Notes:
                - Created 2 because some models may perform better on standardised data than normalised data and vice versa
                - Min-max normalisation preserves relative relationships between data points but eliminates differences in magnitude
                - Standardisation brings data features onto a similar scale to be comparable (Helps remove the influence of the mean and scale of data where distribution of data is not Gaussian or contains outliers)
                """
                DATA[hyperparameters["cols_to_alter"]] = transformation(dataframe = DATA, cols_to_alter = hyperparameters["cols_to_alter"])
            
            # Add this companies data to the list
            self.data.append(self.dataframe_to_ptt(pandas_dataframe = DATA, desired_dtype = torch_float_32))

            # Add the dates to the list
            self.dates.append(DATA.index.tolist())

            print(f"Ticker: {ticker} | DataShape: {self.data[-1].shape} | LabelsShape: {self.labels[-1].shape} | SentimentsShape: {self.single_sentiments[-1].shape}")
        
        print(f"Total examples: {sum([company_labels.shape[0] for company_labels in self.labels])}")

        # Standardising / Normalising companies together
        if hyperparameters["transform_after"] == True:

            # Find indexes of all the columns we want to alter
            col_indexes = [DATA.columns.get_loc(column_name) for column_name in hyperparameters["cols_to_alter"]]

            # Combine all of the companies data, and select the only the column features that we want to alter
            combined_data = torch_cat(self.data, dim = 0)[:, col_indexes]

            # Create standardised or normalised versions of the data 
            transformation(combined_data = combined_data, col_indexes = col_indexes, params_from_training = hyperparameters["train_data_params"])

        # Set the number of features that will go into the first layer of a model
        self.n_features = self.data[-1].shape[1]
        print(f"Number of data features: {self.n_features}")

        # Remove invalid tickers
        for invalid_ticker in invalid_tickers:
            tickers.remove(invalid_ticker)

    def modify_data(self, D, dated_sentiments = None, include_date_before_prediction_date = False, hyperparameters = None):
        
        # Dated sentiments were provided
        if type(dated_sentiments) != type(None):
            # Create new column for dates for merging
            D["post_date"] = D.index.strftime('%Y-%m-%d')
            dated_sentiments.rename(columns = {"ticker_symbol": "ticker"}, inplace = True) # Rename from "ticker_symbol" to "ticker" for merging
            print(dated_sentiments.shape)
            print("Before", D.shape)

            # Merge the two datasets 
            """ Removes all the rows in the DATA dataframe where the combination of "post_date" and "ticker" do not exist in dated_sentiments """
            D = D.merge(dated_sentiments, on = ["post_date", "ticker"])

            # Set the index to be the dates of each row (As after merging, they will become indexes)
            # - Also removes the "post_date" column
            D.set_index("post_date", inplace = True, drop = True) 

            print("After", D.shape, D.columns)
            # for ticker, post_date, sentiment in zip(DATA["ticker"].to_list(), DATA["post_date"].to_list(), DATA["sentiment"].to_list()):
            #     print(ticker, post_date, sentiment)

        # Remove ticker column (always done)
        if "ticker" in D.columns:
            D.drop("ticker", axis = 1, inplace = True)

        # Remove additional un-needed features (done at training or inference)
        if hyperparameters != None:
            d_columns = D.columns
            for r_feature in hyperparameters["features_to_remove"]:
                if r_feature in d_columns:
                    D.drop(r_feature, axis = 1, inplace = True)

        # Create new column for each day stating tomorrow's closing price for the stock
        D["TomorrowClose"] = D["close"].shift(-1)

        # Set the target as 1 if the price of the stock increases tomorrow, otherwise set it to 0 (1 = Price went up, 0 = Price stayed the same / went down)
        D["Target"] = (D["TomorrowClose"] > D["close"]).astype(int)

        # Adding more features
        if hyperparameters != None:
            """
            - 5 days = 1 trading week 
            - Rolling features = _ over the last p days/weeks/months
            - if shift(1) is used, it is not including the current day, otherwise it is
            """
            rolling_features = set(hyperparameters["rolling_features"])
            for p in hyperparameters["rolling_periods"]:

                # Average opening price 
                rolling_open_averages =  D["open"].rolling(window = p).mean()  
                if "avg_open" in rolling_features:
                    D[f"AvgOpen_{p}"] = rolling_open_averages
                
                # Opening price ratio
                if "open_ratio" in rolling_features:
                    D[f"OpenRatio_{p}"] = D["open"] / rolling_open_averages

                # Average closing price 
                rolling_close_averages =  D["close"].rolling(window = p).mean()  
                if "avg_close" in rolling_features:
                    D[f"AvgClose_{p}"] = rolling_close_averages
                
                # Closing price ratio
                if "close_ratio" in rolling_features:
                    D[f"CloseRatio_{p}"] = D["close"] / rolling_close_averages

                # Average volume
                rolling_volume_averages =  D["volume"].rolling(window = p).mean()  
                if "avg_volume" in rolling_features:
                    D[f"AvgVolume_{p}"] = rolling_volume_averages

                # Volume ratio
                if "volume_ratio" in rolling_features:
                    D[f"VolumeRatio_{p}"] = D["volume"] / rolling_volume_averages

                # Trend sum
                rolling_trends = D["Target"].shift(1).rolling(window = p)
                if "trend_sum" in rolling_features:
                    D[f"TrendSum_{p}"] = rolling_trends.sum()
                
                # Trend mean
                if "trend_mean" in rolling_features:
                    D[f"TrendMean_{p}"] = rolling_trends.mean()

            # Single sentiments (Used to extract the sentiment values from the dates that the model is predicting the stock trend on)
            # - include_date_before_prediction_date == False ensures that this isn't code isn't performed at inference
            if hyperparameters["uses_single_sentiments"] == True and include_date_before_prediction_date == False:
                D["sentiment_tomorrow"] = D["sentiment"].shift(-1)
                D.drop("sentiment", axis = 1, inplace = True) # Remove sentiments as a input feature in each row (Replaced with a single sentiment value from the day to predict)
        
        """ Note: 
        - Includes the date used before the date to predict, this is specifically for inference where we want to predict the next day using the previous day (without this, today would be removed from the dataframe)
        """
        if include_date_before_prediction_date:
            # Set as a placeholder value (currently the closing price of today)
            dates = D.index
            D.loc[dates[-1], "TomorrowClose"] = D.loc[dates[-1], "close"] 

        # Removes rows which contain "NaN" inside of any columns
        D.dropna(inplace = True)

        # print(D[:25])

        # Remove "TomorrowClose" as the model shouldn't "know" what tomorrow's closing price is
        D.drop("TomorrowClose", axis = 1, inplace = True)

        # Set all columns in the D to the float datatype (All values must be homogenous when passed as a tensor into a model) and return the dataframe
        return D.astype(float)

    def normalise_columns(self, dataframe, cols_to_alter):
        # Return alteralised columns
        return (dataframe[cols_to_alter] - dataframe[cols_to_alter].min()) / (dataframe[cols_to_alter].max() - dataframe[cols_to_alter].min())
    
    def standardise_columns(self, dataframe, cols_to_alter):
        # Return alterised columns
        return (dataframe[cols_to_alter] - dataframe[cols_to_alter].mean()) / dataframe[cols_to_alter].std()
    
    def standardise_data(self, combined_data, col_indexes, params_from_training = None):

        # Find the mean and std of all the companies across all of the columns to alter 
        if params_from_training == None:
            if hasattr(self, "train_data_params") == False:
                self.train_data_params = {
                                        "S": {
                                            "mean": combined_data.mean(dim = 0),
                                            "std": combined_data.std(dim = 0)
                                             }
                                        }
            else:
                self.train_data_params["S"] = {}
                self.train_data_params["S"]["mean"] = combined_data.mean(dim = 0)
                self.train_data_params["S"]["std"] = combined_data.std(dim = 0)
            
        # At inference time, will use the training parameters saved (self.train_data_params won't exist in the DataHandler in inference)
        # If training, will use the training parameters just created
        params_used = params_from_training if params_from_training != None else self.train_data_params

        # Standardise all the companies together
        for i in range(len(self.data)): 
            self.data[i][:,col_indexes] -= params_used["S"]["mean"]
            self.data[i][:,col_indexes] /= params_used["S"]["std"]
    
    def normalise_data(self, combined_data, col_indexes, params_from_training = None):
        
        # Find the minimums and maximums of all of the companies, for each column (and the difference between them) 
        if params_from_training == None:
            if hasattr(self, "train_data_params") == False:
                self.train_data_params = {
                                        "N":{
                                            "minimums": combined_data.min(dim = 0)[0],
                                            "maximums": combined_data.max(dim = 0)[0],
                                            }
                                        }
            else:
                self.train_data_params["N"] = {}
                self.train_data_params["N"]["minimums"] = combined_data.min(dim = 0)[0]
                self.train_data_params["N"]["maximums"] = combined_data.max(dim = 0)[0]

            self.train_data_params["N"]["differences"] = self.train_data_params["N"]["maximums"] - self.train_data_params["N"]["minimums"]
        
        # At inference time, will use the training parameters saved (self.train_data_params won't exist in the DataHandler in inference)
        # If training, will use the training parameters just created
        params_used = params_from_training if params_from_training != None else self.train_data_params

        # Normalise all the companies together
        for i in range(len(self.data)):
            self.data[i][:, col_indexes] -= params_used["N"]["minimums"]
            self.data[i][:, col_indexes] /= params_used["N"]["differences"]

    def dataframe_to_ptt(self, pandas_dataframe, desired_dtype = torch_float_32):
        
        # Converts a pandas dataframe to a PyTorch tensor (With the default dtype being torch.float32)
        # 1. pandas_dataframe.values extracts the data into numpy arrays (dtype = float64) [Maintains shape]
        # 2. torch_tensor converts np array to PyTorch tensor

        return torch_tensor(pandas_dataframe.values, dtype = desired_dtype)

    def generate_batch(self, batch_size, dataset, num_context_days, start_idx, uses_single_sentiments):
        
        # Find the inputs and labels (and sentiments) in the dataset and find the number of examples in this set

        if uses_single_sentiments:
            inputs, labels, sentiments = dataset
        else:
            inputs, labels = dataset

        # Generate batch indexes starting from the start index, which correspond to each example in the labels and inputs of this dataset (perform using CUDA if possible)
        example_idxs = [start_idx + idx for idx in range(batch_size)]

        if num_context_days == 1:
            # Return the examples and the corresponding targets (for predicting whether the price goes up or down for the next day) (+ sentiments)
            # Note: If self.device == "cuda", then the batch will be moved back onto the GPU
            return inputs[example_idxs].to(device = self.device), labels[example_idxs].to(device = self.device), sentiments[example_idxs].to(device = self.device) if uses_single_sentiments else None

        else:
            # Notes:
            # - num_examples = Number of "num_context_days" sequences
            # - Batch should be of shape [num_context_days, batch_size, sequence length] (So each day will have e.g. 32 examples)
            # - For the labels, the target for each day in the context day should be the target of the last day in the sequence (i.e labels[example_idx][-1])
            # - In other words, with the context of e.g. 10 days, we are trying to predict the trend for the 11th day

            # # Each day in "num_context_days" should have "batch_size" sequences
            # b_inputs = []
            # for i in range(num_context_days):
            #     # i = The current day in the sequence, j = the index inside the example_idxs for the batch
            #     current_day_I = torch_stack([inputs[example_idxs[j]][i] for j in range(batch_size)], dim = 0)
            #     b_inputs.append(current_day_I)
            # b_inputs = torch_stack(b_inputs, dim = 0)
            # b_labels = torch_stack([labels[example_idx][-1] for example_idx in example_idxs])

            # Each day in "num_context_days" should have "batch_size" sequences (Performs the same code as above)
            b_inputs = torch_stack(tensors = [torch_stack([inputs[example_idxs[j]][i] for j in range(batch_size)], dim = 0) for i in range(num_context_days)], dim = 0)
            
            # Return the batch inputs, labels and sentiments
            return b_inputs.to(device = self.device), labels[example_idxs].to(device = self.device), sentiments[example_idxs].to(device = self.device) if uses_single_sentiments else None

    def create_data_sequences(self, num_context_days, shuffle_data_sequences):
        # Converts self.data, self.labels into data sequences

        # MLP 
        if num_context_days == 1:
            
            # Combine all companies data together
            # data.shape = [number of single day examples, number of features for each day]
            # labels.shape = [number of single day examples, correct prediction for stock trend for the following day]
            
            all_data = []
            all_labels = []
            all_dates = []
            self.sequence_sizes = []

            for x, (c_labels, c_dates, c_data) in enumerate(zip(self.labels, self.dates, self.data)):
                # Add the data. labels and dates for this company to the lists
                all_data.append(c_data)
                all_labels.append(c_labels)
                all_dates.extend(c_dates)

                # Add the number of sequences for this company
                self.sequence_sizes.append(c_labels.shape[0])

                print(f"Company {x} | LabelsShape {c_labels.shape} | DataShape {c_data.shape}")

            # Concatenate all the data and labels from all the companies
            self.data = torch_cat(all_data, dim = 0)
            self.labels = torch_cat(all_labels, dim = 0)
            self.dates = all_dates

        # RNN
        else:
            
            # Create all the data sequences from each company
            all_data_sequences = []
            all_labels = []
            all_dates = []
            self.sequence_sizes = [] # Used to find out which sequences belong to which companies (at inference time)

            # Data (Normalised and standardised versions)
            for x, (c_labels, c_dates, c_data) in enumerate(zip(self.labels, self.dates, self.data)):

                # The number of sequences of length "num_context_days" in this company's data
                original_num_days = c_labels.shape[0] # Original number of days
                num_sequences = original_num_days - (num_context_days - 1) # E.g. if num_context_days = 10, 4530 --> (4530 - (10 - 1)) = 4519
                start_trim_idx = original_num_days - num_sequences

                # Trim labels (The same is done for self.data when converting to sequences)
                """
                Dates = [Jan1, Jan2, Jan3, Jan4, Jan5, Jan6, Jan7, Jan8, Jan9, Jan10, Jan11....]
                start_trim_idx = 9
                Dates = dates[start_trim_idx:] --> [Jan11 ....] (CORRECT)

                           0     1     2     3     4     5      6    7     8     9      10
                labels = [Jan1, Jan2, Jan3, Jan4, Jan5, Jan6, Jan7, Jan8, Jan9, Jan10, Jan11....]

                First sequence = Jan1, Jan2, Jan3, Jan4, Jan5, Jan6, Jan7, Jan8, Jan9, Jan10
                Correct label for January 10 predicting January 11 is labels[9]

                Second sequence = Jan2, Jan3, Jan4, Jan5, Jan6, Jan7, Jan8, Jan9, Jan10, Jan11
                Correct label for January 11 predicting January 12 is labels[10]
                """
                c_labels = c_labels[start_trim_idx:] # labels = (Correct predictions for all data sequences in self.data)

                # Trim dates 
                # - Each c_data[i] should correspond to the correct date in c_dates[i], where c_dates[i] is the date before the date to predict
                c_dates = c_dates[start_trim_idx:]

                # Let num_context_days = 10, batch_size = 32
                # Single batch should be [10 x [32 * num_features] ]
                # 32 x [ClosingP, OpeningP, Volume, etc..] 10 days ago
                # The next batch for the recurrence will be the day after that day
                # 32 x [ClosingP, OpeningP, Volume, etc..] 9 days ago
                # Repeats until all 10 days have been passed in (for a single batch)
                # c_data.shape = (number of 10 consecutive days sequences, 10 consecutive days of examples, number of features in each day)
                c_data = torch_stack([c_data[i:i + num_context_days] for i in range(0, num_sequences)], dim = 0)

                # Add the data sequences and labels for this company to the lists
                all_data_sequences.append(c_data)

                # Add the labels for this company to the lists
                all_labels.append(c_labels)

                # Add the dates for this company to the list
                all_dates.extend(c_dates) # Extend so that it is a single list

                # Add the number of sequences for this company
                self.sequence_sizes.append(c_labels.shape[0])

                print(f"Company {x} | LabelsShape {c_labels.shape} | DataShape {c_data.shape}")

            # Concatenate all the data sequences and labels from all the companies
            self.data = torch_cat(all_data_sequences, dim = 0)
            self.labels = torch_cat(all_labels, dim = 0)
            self.dates = all_dates

        # Using single sentiment values for model (instead of each data sequence having a sentiment)
        if hasattr(self, "single_sentiments"):

            if num_context_days == 1:
                all_sentiments = [c_sentiments for c_sentiments in self.single_sentiments]
            else:
                # Trim the sentiments
                # - sentiments = (The sentiment values on the day to predict)
                all_sentiments = [c_sentiments[start_trim_idx:] for c_sentiments in self.single_sentiments] 
            
            # Concatenate all companies sentiments
            self.single_sentiments = torch_cat(all_sentiments, dim = 0)

            # Convert to a 2D tensor, as it will need to be concatenated with the final state of the inputs of the models before entering the output layer
            self.single_sentiments = self.single_sentiments.view(self.single_sentiments.shape[0], 1)

        # Sorting / Shuffling data sequences
        if shuffle_data_sequences == True:
            self.shuffle_data_sequences() # Random shuffle
        else:
            self.sort_data_sequences(dates = self.dates) # Chronological order
        
        # Remove dates (no longer required)
        # del self.dates
    
        print(f"DataShape: {self.data.shape} | LabelsShape: {self.labels.shape} " + f"| SingleSentiments: {self.single_sentiments.shape}" if hasattr(self, "single_sentiments") else "")

    def separate_data_sequences(self, train_split_decimal):
        # Separates data sequences into training and test sets 
        # - The training set will be used for folds during training
        # - The test set will be used as a final evaluation of then model after cross-validation

        # Predicting with single sentiments
        if hasattr(self, "single_sentiments"):
            train_end_idx = int(self.labels.shape[0] * train_split_decimal)
            self.TRAIN_SET = (self.data[0:train_end_idx], self.labels[0:train_end_idx], self.single_sentiments[0:train_end_idx])
            self.TEST_SET = (self.data[train_end_idx:], self.labels[train_end_idx:], self.single_sentiments[train_end_idx:])

            print(f"TRAIN SET | Inputs: {self.TRAIN_SET[0].shape} | Labels: {self.TRAIN_SET[1].shape} | SingleSentiments: {self.TRAIN_SET[2].shape}")
            print(f"TEST SET | Inputs: {self.TEST_SET[0].shape} | Labels: {self.TEST_SET[1].shape} | SingleSentiments: {self.TEST_SET[2].shape}")

        # Without single sentiments
        else:
            train_end_idx = int(self.labels.shape[0] * train_split_decimal)
            self.TRAIN_SET = (self.data[0:train_end_idx], self.labels[0:train_end_idx])
            self.TEST_SET = (self.data[train_end_idx:], self.labels[train_end_idx:])
            
            print(f"TRAIN SET | Inputs: {self.TRAIN_SET[0].shape} | Labels: {self.TRAIN_SET[1].shape}")
            print(f"TEST SET | Inputs: {self.TEST_SET[0].shape} | Labels: {self.TEST_SET[1].shape}")
    
    def shuffle_data_sequences(self):
        # Shuffling the data sequences

        permutation_indices = torch_randperm(self.labels.shape[0], device = self.device, generator = self.generator) # Generate random permutation of indices
        permutation_indices = permutation_indices.to(device = "cpu") # Move to CPU as self.data is on the CPU

        # Assign indices to data, labels (and sentiments)
        self.data = self.data[permutation_indices]
        self.labels = self.labels[permutation_indices]
        if hasattr(self, "single_sentiments"):
            self.single_sentiments = self.single_sentiments[permutation_indices]
    
    def sort_data_sequences(self, dates):
        
        # Convert the list of pandas timestamps into unix timestamps (Number of seconds that have elapsed since January 1, 1970 (UTC))
        # Notes: 
        # - Converted to unix timestamps so that we can perform torch.argsort()
        # - self.data and self.labels at this stage will be the companies data placed one after another, so the dates must be sorted into chronological order
        # - descending = False because the timestamps are seconds elapsed so larger numbers are further in the future
        unix_timestamps = torch_tensor([pd_to_datetime(time_stamp).timestamp() for time_stamp in dates])
        sort_indices = torch_argsort(unix_timestamps, descending = False) # Returns the indices which will sort self.data and self.labels in chronological order 

        #print("SORTED: DATALABELSDATES", self.data.shape, self.labels.shape, sort_indices.shape)
        self.sort_indices = sort_indices
        # Sort data, labels, dates (and sentiments)
        self.data = self.data[sort_indices]
        self.labels = self.labels[sort_indices] 
        self.dates = [self.dates[i] for i in sort_indices]
        if hasattr(self, "single_sentiments"):
            self.single_sentiments = self.single_sentiments[sort_indices]

    def create_sets(self, num_context_days, shuffle_data_sequences, train_split_decimal):
        # Convert self.data, self.labels into data sequences 
        self.create_data_sequences(num_context_days = num_context_days, shuffle_data_sequences = shuffle_data_sequences)

        # Separate the data sequences into to two sets (Training and test)
        self.separate_data_sequences(train_split_decimal = train_split_decimal)

    def create_folds(self, num_folds):
        # Creates folds out of the training set

        # Divide the data and labels into k folds
        # Note: self.d_folds and self.l_folds will be a tuple of k folds, with each fold being a tensor
        # setattr(self, f"d_folds", torch_chunk(input = self.TRAIN_SET[0], chunks = num_folds, dim = 0))
        # setattr(self, f"l_folds", torch_chunk(input = self.TRAIN_SET[1], chunks = num_folds, dim = 0))

        self.D_FOLDS = torch_chunk(input = self.TRAIN_SET[0], chunks = num_folds, dim = 0)
        self.L_FOLDS = torch_chunk(input = self.TRAIN_SET[1], chunks = num_folds, dim = 0)
        if hasattr(self, "single_sentiments"):
            self.S_FOLDS = torch_chunk(input = self.TRAIN_SET[2], chunks = num_folds, dim = 0)

    def retrieve_k_folds(self, window_size, use_single_sentiments):
        
        # Implementation of walk-forward / expanding window cross validation:
        # = Selects the fold at the end of the window as the validation set and the remaining folds for training (k will be zero indexed)
        # - Models will always be trained on past data with the validation set being new unseen data.

        if use_single_sentiments:
            # Retrieve only the first "window_size" folds
            D_FOLDS = self.D_FOLDS[:window_size]
            L_FOLDS = self.L_FOLDS[:window_size]
            S_FOLDS = self.S_FOLDS[:window_size]
            
            # Example:
            # Train: [Fold1], Validation: Fold2
            # Train: [Fold1, Fold2], Validation: Fold3
            # Train: [Fold1, Fold2, Fold3], Validation: Fold4
            # Train: [Fold1, Fold2, Fold3, Fold4], Validation: Fold5

            print(f"TRAIN FOLDS | Inputs: {torch_cat([D_FOLDS[i] for i in range(window_size - 1)]).shape} | Labels: {torch_cat([L_FOLDS[i] for i in range(window_size - 1)]).shape} | SingleSentiments: {torch_cat([S_FOLDS[i] for i in range(window_size - 1)]).shape}")
            print(f"VAL FOLD | Inputs: {D_FOLDS[-1].shape} | Labels: {L_FOLDS[-1].shape} | SingleSentiments: {S_FOLDS[-1].shape}")


            # Training folds and validation fold
            # Note: Each tuple = (data folds, corresponding label folds)
            # Return the training folds and validation folds
            return (
                    torch_cat([D_FOLDS[i] for i in range(window_size - 1)]), 
                    torch_cat([L_FOLDS[i] for i in range(window_size - 1)]),
                    torch_cat([S_FOLDS[i] for i in range(window_size - 1)])
                    ), (D_FOLDS[-1], L_FOLDS[-1], S_FOLDS[-1])

        else:

            # Retrieve only the first "window_size" folds
            D_FOLDS = self.D_FOLDS[:window_size]
            L_FOLDS = self.L_FOLDS[:window_size]
            
            # Example:
            # Train: [Fold1], Validation: Fold2
            # Train: [Fold1, Fold2], Validation: Fold3
            # Train: [Fold1, Fold2, Fold3], Validation: Fold4
            # Train: [Fold1, Fold2, Fold3, Fold4], Validation: Fold5
            print(f"TRAIN FOLDS | Inputs: {torch_cat([D_FOLDS[i] for i in range(window_size - 1)]).shape} | Labels: {torch_cat([L_FOLDS[i] for i in range(window_size - 1)]).shape}")
            print(f"VAL FOLD | Inputs: {D_FOLDS[-1].shape} | Labels: {L_FOLDS[-1].shape}")

            # Training folds and validation fold
            # Note: Each tuple = (data folds, corresponding label folds)
            # Return the training folds  and validation folds
            return (torch_cat([D_FOLDS[i] for i in range(window_size - 1)]), torch_cat([L_FOLDS[i] for i in range(window_size - 1)])), (D_FOLDS[-1], L_FOLDS[-1])
        
class TextDataHandler: 
    """ Will be used to train my own sentiment analysis model after using a pretrained model to label the unlabelled dataset with sentiment values """

    def __init__(self, dates, device, generator):

        self.device = device # CUDA or CPU
        self.generator = generator # Reproducibility

        # Dates from all the companies that appear in the historical dataset (Combine into a single list of all the dates)
        # - timestamp.date() to convert the timestamps into the correct format before removing tweets that don't also appear in the historical dataset
        self.dates = [timestamp.date() for company_dates in dates for timestamp in company_dates] 
    
    def retrieve_data(self):
        
        # Note:
        # - Contains data on the top companies from January 1st 2015 - December 31st 2019 (Inclusive)
        # - The reason why MERGED.shape is larger when DATA.shape is smaller than TWEET_COMPANY.shape is because a single tweet (in DATA) can reference multiple different companies

        # Labeled data already created
        if os_path_exists("sentiment_data/progress/labeled_tweets.csv"):
            print("Loading labeled tweets")
            LABELED_TWEETS = pd_read_csv("sentiment_data/progress/labeled_tweets.csv")

        # Not created yet
        else:
            print("Labeling tweets")
            # Data containing the tweet id, writer, post date, tweet, number of comments, likes and retweets
            DATA = pd_read_csv("sentiment_data/Tweet.csv")

            # The IDs to all the tweets and what company they were referring to
            TWEET_COMPANY = pd_read_csv("sentiment_data/Company_Tweet.csv")

            # Remove any tweets if it was posted on dates that do not appear inside the historical dataset
            """ 
            - pd_date_time(__, unit = "s") to convert unix timestamps to dates
            - .dt.date to round the dates from e.g. 2015-01-01 00:01:36 to 2015-01-01 00:00:00
            """
            # print(DATA)
            DATA["post_date"] = pd_to_datetime(DATA["post_date"], unit = "s").dt.date # Convert unix timestamps to dates in the data column
            DATA = DATA[DATA["post_date"].isin(self.dates)]
            # print(DATA)

            # Merge the company tickers with each tweet according to tweet id
            MERGED = DATA.merge(TWEET_COMPANY, on = "tweet_id", how = "left")

            print(MERGED[MERGED["body"] == ""])

            # print(DATA[DATA.isna().any(axis = 1)])
            # print(TWEET_COMPANY[TWEET_COMPANY.isna().any(axis = 1)])
            # print(MERGED[MERGED.isna().any(axis = 1)])

            print(MERGED[MERGED["body"].isna()])

            # Label the merged dataset with sentiment values
            LABELED_TWEETS = self.label_dataset(dataset = MERGED)

        print(LABELED_TWEETS)
        
        # Find the mean sentiment value for each company on a given date
        # mean = []
        # for post_date, ticker_symbol, sentiment in zip(LABELED_TWEETS["post_date"].to_list(), LABELED_TWEETS["ticker_symbol"].to_list(), LABELED_TWEETS["sentiment"].to_list()):
        #     if post_date != "2016-12-27":
        #         break
        #     if ticker_symbol != "TSLA":
        #         continue
        #     mean.append(sentiment)
        #     # print(post_date, ticker_symbol, sentiment)
        # mean = sum(mean) / len(mean)
        # print(mean)
        """ 
        - .mean() to find the mean sentiment value
        - .reset_index() to convert back into a dataframe
        - "body" column removed as it is no longer necessary, and so that .mean() will return the average sentiment for each date for each company
        """
        LABELED_TWEETS.drop("body", axis = 1, inplace = True)
        self.dated_sentiments = LABELED_TWEETS.groupby(["post_date", "ticker_symbol"]).mean().reset_index() # Group the dateset by the dates and ticker symbols
        # print(self.dated_sentiments)
        # print(self.dated_sentiments["sentiment"].to_list()[:6])

    def clean_dataset(self, dataset):
        
        from re import sub as re_sub

        stop_words = set([
                        "the",
                        "and", 
                        "in",
                        ]
                        )

        def cleanse_text(text):
            # Removes links, mentions, multiple spaces, hashtags and sets the text to lowercase
            
            # Regular expression pattern to match URLs
            url_pattern = r"http\S+"
            # Regular expression pattern to match mentions
            mention_pattern = r"@\w+"
            # Regular expression pattern to match multiple spaces
            multiple_spaces_pattern = r"\s+"
            # # Regular expression pattern to match multiple spaces
            # hashtag_pattern = r"#\w+"
            # Regular expression pattern to match non-utf 8/ non ASCII character
            character_type_pattern = r"[^\x00-\x7f]"

            # Remove \r and \n (New lines) and return the string in lowercase
            clean_text = text.replace("\r", "").replace("\n", "").lower()

            # Remove URLs
            clean_text = re_sub(url_pattern, "", clean_text)

            ## Remove hashtags 
            # Note: (Might not ideal when users use e.g. "#google" near the start or middle of the text, instead of at the end)
            # clean_text = re_sub(hashtag_pattern, "", clean_text)

            # Remove characters that are not utf-8 / ascii
            clean_text = re_sub(character_type_pattern, "", clean_text)

            # Remove mentions
            clean_text = re_sub(mention_pattern, "", clean_text)
            
            # Clean any remaining "@" [This happens when a "@" comes after a word without a space]
            clean_text = clean_text.replace("@", "")

            # Remove all "|" as this will be used as our delimiter when reading / writing the saved clean dataset
            clean_text = clean_text.replace("|", "")

            # Remove '" "' in text (Appeared in the dataset a few times)
            clean_text = clean_text.replace('" "', "")

            # Remove multiple spaces (replaced with a single space)
            # Note: Do this after removing URLs, mentions and hashtags (as they could create new gaps)
            clean_text = re_sub(multiple_spaces_pattern, " ", clean_text)

            # Remove stop words (commonly occurring words which provide insignificant value to the model)
            clean_text = clean_text.split()
            clean_text = " ".join(word for word in clean_text if word not in stop_words)

            # Remove leading exclamation marks, semi colons, commas, spaces 
            # Note: (Did not remove full stops because users may have used ellipses)
            clean_text = clean_text.lstrip(" :!,")
            clean_text = clean_text.rstrip(": ")

            return clean_text
        
        def format_text(tweet, ticker):
            # Formats the relevant information into a desired prompt format, which will be passed into the model
            # - Format is from the llama-cpp-python library example
            return f"Q: Assign a sentiment for the text which is relevant to the company with the '{ticker}' ticker (-1=Negative, 0=Neutral, 1=Positive, 2=Not relevant), given the text '{tweet}'. Only respond with a value. A:"
            # return f"""Assign a sentiment(-1=Negative, 0=Neutral, 1=Positive) for the company with the '{ticker}' ticker, given the text '{tweet}'. For example, in a different scenario, if the text was 'Tesla is great' and the ticker was "tsla", your answer should be 1 (Positive sentiment) as the text talks positively about Tesla, who has the "tsla" ticker. If the tweet is not relevant to the company, your answer should be 0"""

        print("VALUE COUNTS")
        value_counts = dataset[["body", "ticker_symbol"]].value_counts()
        print(value_counts)
        print(dataset[["body", "ticker_symbol"]].nunique())
        print("Unique combinations", len(value_counts), "Dataset size", len(dataset))

        # for tweet, ticker in zip(dataset["body"], dataset["ticker_symbol"]):
        #     print("!", tweet, f"Ticker: {ticker}"))
        print(f"Dataset size before: {dataset.shape}")
        # Apply cleaning to all text
        dataset["body"] = dataset["body"].apply(lambda x: cleanse_text(x))
        print(f"Dataset size after: {dataset.shape}")

        # Remove any rows where the tweet is now empty (Prevents prompts with empty texts being used)
        dataset.drop(dataset[dataset["body"] == ""].index, inplace = True)

        # # Create a new column containing the desired prompt with the text and ticker symbol inserted for each row
        # dataset["prompt"] = dataset.apply(lambda row: format_text(tweet = row["body"], ticker = row["ticker_symbol"]), axis = 1)

        # print(dataset[dataset["body"].isna()])
        # print(dataset[dataset["body"] == ""])
        # print(dataset[dataset["body"] == ""].index)

        print("CLEANED + FORMATTED")
        value_counts = dataset[["body", "ticker_symbol"]].value_counts()
        print(value_counts)
        print(dataset[["body", "ticker_symbol"]].nunique())
        print("Unique combinations", len(value_counts), "Dataset size", len(dataset))
        
        # for tweet, ticker in zip(dataset["body"], dataset["ticker_symbol"]):
        #     print("!", tweet, f"Ticker: {ticker}")

        """ 
        Notes:

        unique_text = "Amazon is great", ticker_symbol = "amzn" should count as a single unique instance, 
        - If the same text appeared again but with a different ticker, that should be a different instance
        - Get the sentiment scores on all the combinations and store them in a different dataset
        - Merge that dataset with the dataset containing all of the tweets based on the tweet and the ticker symbol
        """
        return dataset
    
    def label_dataset(self, dataset):
        # Labels the unlabeled dataset (containing the tweets about the top companies from 2015 to 2020)

        print(dataset.columns)

        clean_data_path = "sentiment_data/progress/cleaned_data.csv"
        custom_separator = "|"
        from csv import QUOTE_NONE as csv_QUOTE_NONE
        if os_path_exists(clean_data_path):
            print("Loading clean data")
            # Load the cleaned data
            all_tweets = pd_read_csv(clean_data_path, sep = custom_separator, quoting = 3)

            print(all_tweets[all_tweets["body"] == ""].index)

        else:
            print("Creating cleaned data")
            # Create a copy of all the tweets and put them in a Python list
            all_tweets = dataset[["writer", "post_date", "body", "ticker_symbol"]].copy()

            print(all_tweets)

            # Create a clean and formatted version of the all_tweets dataset
            all_tweets = self.clean_dataset(dataset = all_tweets)
            all_tweets.drop(["writer"], axis = 1, inplace = True) # Drop unrequired columns

            # Create progress directory if it does not exist
            progress_path = "sentiment_data/progress"
            if os_path_exists(progress_path) == False:
                os_mkdir("sentiment_data/progress")

            # Save cleaned data as a csv file
            # Note: csv.QUOTE_NONE so that quotation marks aren't added around the tweets / prompts, custom separator used as delimeter as commas are used in texts
            all_tweets.to_csv(clean_data_path, index = False, sep = custom_separator, quoting = csv_QUOTE_NONE)
            
        
        # Create a prompt dataset
        """
        Notes:
        - Remove all duplicates based on the tweet and selected ticker symbol (for entity-based sentiment analysis)
        - Will be used for inference of the model
        - groupby groups the dataframe by the selected columns
        - as_index = False to ensure that the dataframe uses a regular index instead of multiindex
        """
        prompt_dataset = all_tweets[["body", "ticker_symbol"]].groupby(["body", "ticker_symbol"], as_index = False).first()

        print("REMOVED DUPLICATES")
        value_counts = prompt_dataset[["body", "ticker_symbol"]].value_counts()
        print(value_counts)
        print(prompt_dataset[["body", "ticker_symbol"]].nunique())
        print("Unique combinations", len(value_counts), "prompt_dataset size", len(prompt_dataset))
        print(prompt_dataset[prompt_dataset["body"] == ""])
        print(prompt_dataset.columns)

        print(all_tweets)
        print(prompt_dataset)

        # ---------------------------------------------------------------------------
        # Build a new column to store the predictions from the model

        """
        Notes: 
        - The returned outputs will be a single integer. [-1 for negative, 0 for neutral, 1 for positive, 2 for unrelated]
        """
        checkpoints_path = "sentiment_data/progress/sentiment_cps"

        if os_path_exists(checkpoints_path) == False:
            print("Querying model")
            os_mkdir(checkpoints_path)

            # Use VADER model to label dataset
            # Note: Considered using FinVADER but inference times take significantly longer and it isn't significantly better at assigning sentiments correctly than VADER
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            from pickle import dump as pickle_dump
            from numpy import array_split as np_array_split

            # Split the tweets into parts and save the scores assigned after each part when querying the model (Checkpoints)
            num_cpoints = 10
            tweets_to_label = prompt_dataset["body"].to_list()
            tweet_parts = np_array_split(tweets_to_label, num_cpoints)
            del tweets_to_label

            for part in tweet_parts:
                print(len(part))

            # Generate responses from VADER
            vader = SentimentIntensityAnalyzer()
            for i, part in enumerate(tweet_parts):
                start_time = get_time()
                sentiments = []
                # Label tweets in this part
                for tweet in part:
                    sentiments.append(vader.polarity_scores(tweet)["compound"])
                end_time = get_time()

                # Save checkpoint (Reset the list for the next section of tweets_to_label)
                with open(os_path_join(checkpoints_path, f"sentiments_cp{i}"), "wb") as f:
                    print("Length", len(sentiments))
                    pickle_dump(sentiments, f)
                
                print(f"Part {i + 1} / {num_cpoints} | Time taken: {end_time - start_time}")
        
        # Combine all sentiments from each checkpoint into a single list
        from pickle import load as pickle_load
        from os import listdir as os_list_dir
        print("Combining sentiments")
        print(os_list_dir(checkpoints_path))
        all_sentiments = []
        for sentiment_cp_fname in os_list_dir(checkpoints_path):
            with open(os_path_join(checkpoints_path, sentiment_cp_fname), "rb") as sf:
                sentiments = pickle_load(sf)
            all_sentiments.extend(sentiments)
        
        print(all_sentiments[-50:])
        print(len(all_sentiments))

        # Create new column with the assigned sentiments for each tweet
        prompt_dataset["sentiment"] = all_sentiments

        print(prompt_dataset.columns)
        print(all_tweets.columns)
        print(prompt_dataset)
        print(all_tweets)
        
        # Map the tweets with the corresponding sentiment scores
        start_time = get_time()
        # Note: Merging on either doesn't make a difference in execution time
        labeled_tweets = all_tweets.merge(prompt_dataset, on = ["body", "ticker_symbol"], how = "left")
        end_time = get_time()

        print(end_time - start_time)

        print(labeled_tweets.columns, len(labeled_tweets))
        print(labeled_tweets)

        # Save the labeled dataset as a csv file
        labeled_tweets.to_csv("sentiment_data/progress/labeled_tweets.csv", index = False)

        # Re-load the labelled tweets and return it
        # - Fixes the issue where if the tweets were labelled from scratch and labeled_tweets was returned, the modify_data method in the DataHandler does not work.
        return pd_read_csv("sentiment_data/progress/labeled_tweets.csv")

