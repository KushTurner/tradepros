from yahoo_fin.stock_info import get_data
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

class DataHandler:

    def __init__(self, device, generator):

        self.device = device # CUDA or CPU
        self.generator = generator # Reproducibility

        # self.data - Holds the inputs for each example
        # self.labels - Holds the corresponding targets for each example
        # self.TRAIN_S(N/S) - Training set
        # self.TEST_S(N/S) - Test set
        
        # self.n_features - Number of inputs that will be passed into a model (i.e. the number of columns/features in the pandas dataframe)

    def retrieve_data(self, tickers, start_date, end_date, interval, transform_after):
        # Note: transform_after = True means that all of the companies' data will be standardised / normalised together, instead of separately
        
        self.data_n = [] # Normalised data
        self.data_s = [] # Standardised data
        self.labels = []
        self.dates = [] # Dates of all the companies (Used to sort the data sequences into chronological order)
        cols_to_alter = ["open", "close", "adjclose", "high", "low", "volume"] # Columns to normalise / standardise
        
        # For each company, modify the data 
        for ticker in tickers:
            
            # Retrieve data
            DATA = get_data(
                            ticker = ticker, 
                            start_date = start_date, 
                            end_date = end_date, 
                            index_as_date = True, 
                            interval = interval
                            )
            # Modify the data (e.g. adding more columns, removing columns, etc.)
            DATA = self.modify_data(D = DATA, interval = interval)
            
            # Separate the labels from the main dataframe (the other columns will be used as inputs)
            labels = DATA["Target"]
            self.labels.append(self.dataframe_to_ptt(pandas_dataframe = labels, desired_dtype = torch_int_64))
            DATA.drop("Target", axis = 1, inplace = True)

            # Create normalised and standardised versions of the data
            S_DATA = DATA.copy()
            if transform_after == False: # Standardising / Normalising companies separately
                # Notes:
                # - Created 2 because some models may perform better on standardised data than normalised data and vice versa
                # - Min-max normalisation preserves relative relationships between data points but eliminates differences in magnitude
                # - Standardisation brings data features onto a similar scale to be comparable (Helps remove the influence of the mean and scale of data where distribution of data is not Gaussian or contains outliers)
                S_DATA[cols_to_alter] = self.standardise_columns(dataframe = S_DATA, cols_to_standard = cols_to_alter)
                DATA[cols_to_alter] = self.normalise_columns(dataframe = DATA, cols_to_norm = cols_to_alter)

            # Add this companies data to the list
            self.data_n.append(self.dataframe_to_ptt(pandas_dataframe = DATA, desired_dtype = torch_float_32))
            self.data_s.append(self.dataframe_to_ptt(pandas_dataframe = S_DATA, desired_dtype = torch_float_32))

            # Add the dates to the list
            self.dates.append(DATA.index.tolist())

            print(f"Ticker: {ticker} | DataShape: {self.data_n[-1].shape} | LabelsShape: {self.labels[-1].shape}")

        # Standardising / Normalising companies together
        if transform_after == True:

            # Find indexes of all the columns we want to alter
            col_indexes = [DATA.columns.get_loc(column_name) for column_name in cols_to_alter]

            # Combine all of the companies data, and select the only the column features that we want to alter
            combined_data = torch_cat(self.data_n, dim = 0)[:, col_indexes]

            # Create standardised + normalised versions of the data
            # print("B", self.data_n[-1][0])
            self.standardise_data(combined_data = combined_data, col_indexes = col_indexes) # Applied to self.data_s
            self.normalise_data(combined_data = combined_data, col_indexes = col_indexes) # Applied to self.data_n
            # print("A", self.data_n[-1][0])

        # Set the number of features that will go into the first layer of a model
        self.n_features = self.data_n[-1].shape[1]
        print(f"Number of data features: {self.n_features}")

    def modify_data(self, D, interval):
        
        # Remove "ticker" column
        D.drop("ticker", axis = 1, inplace = True)

        # Create new column for each day stating tomorrow's closing price for the stock
        D["TomorrowClose"] = D["close"].shift(-1)

        # Set the target as 1 if the price of the stock increases tomorrow, otherwise set it to 0 (1 = Price went up, 0 = Price stayed the same / went down)
        D["Target"] = (D["TomorrowClose"] > D["close"]).astype(int)
        
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
            rolling_averages =  D["close"].rolling(window = p).mean()  

            cr_column_name = f"CloseRatio_{p}"
            # Find the ratio closing price and the average closing price over the last p days/weeks/months (Inclusive of the current day)
            D[cr_column_name] = D["close"] / rolling_averages

            # Trend over the past few days
            t_column_name = f"Trend_{p}"
            D[t_column_name] = D.shift(1).rolling(p).sum()["Target"] # Sums up the targets over the last p days/weeks/months (Not including today's target)

        # Removes rows which contain "NaN" inside of any columns
        D.dropna(inplace = True)

        # Remove "TomorrowClose" as the model shouldn't "know" what tomorrow's closing price is
        D.drop("TomorrowClose", axis = 1, inplace = True)

        # Set all columns in the D to the float datatype (All values must be homogenous when passed as a tensor into a model) and return the dataframe
        return D.astype(float)

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
    
    def standardise_data(self, combined_data, col_indexes):

        # Find the mean and std of all the companies across all of the columns to alter 
        mean = combined_data.mean(dim = 0)
        std = combined_data.std(dim = 0)

        # Standardise for each company
        for i in range(len(self.data_s)): 
            self.data_s[i][:,col_indexes] -= mean
            self.data_s[i][:,col_indexes] /= std
    
    def normalise_data(self, combined_data, col_indexes):

        # Find the minimums and maximums of all of the companies, for each column (and the difference between them) 
        minimums, _ = combined_data.min(dim = 0)
        maximums, _ = combined_data.max(dim = 0)
        differences = maximums - minimums

        # Normalise for each company
        for i in range(len(self.data_n)):
            self.data_n[i][:, col_indexes] -= minimums
            self.data_n[i][:, col_indexes] /= differences

    def dataframe_to_ptt(self, pandas_dataframe, desired_dtype = torch_float_32):
        
        # Converts a pandas dataframe to a PyTorch tensor (With the default dtype being torch.float32)
        # 1. pandas_dataframe.values extracts the data into numpy arrays (dtype = float64) [Maintains shape]
        # 2. torch_tensor converts np array to PyTorch tensor

        return torch_tensor(pandas_dataframe.values, dtype = desired_dtype)

    def generate_batch(self, batch_size, dataset, num_context_days, start_idx):
        
        # Find the inputs and labels in the dataset and find the number of examples in this set
        inputs, labels = dataset

        # Generate batch indexes starting from the start index, which correspond to each example in the labels and inputs of this dataset (perform using CUDA if possible)
        example_idxs = [start_idx + idx for idx in range(batch_size)]

        if num_context_days == 1:
            # Return the examples and the corresponding targets (for predicting whether the price goes up or down for the next day)
            # Note: If self.device == "cuda", then the batch will be moved back onto the GPU
            return inputs[example_idxs].to(device = self.device), labels[example_idxs].to(device = self.device) 

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
            
            # Return the batch inputs and labels
            return b_inputs.to(device = self.device), labels[example_idxs].to(device = self.device)

    def create_data_sequences(self, num_context_days, shuffle_data_sequences):
        # Converts self.data_n, self.data_s, self.labels into data sequences

        # MLP 
        if num_context_days == 1:
            
            # Combine all companies data together
            # data.shape = [number of single day examples, number of features for each day]
            # labels.shape = [number of single day examples, correct prediction for stock trend for the following day]
            
            all_data_n = []
            all_data_s = []
            all_labels = []
            all_dates = []

            for i, (c_labels, c_dates, c_data_n, c_data_s) in enumerate(zip(self.labels, self.dates, self.data_n, self.data_s)):
                # Add the data. labels and dates for this company to the lists
                all_data_n.append(c_data_n)
                all_data_s.append(c_data_s)
                all_labels.append(c_labels)
                all_dates.extend(c_dates)
                print(f"Company {i} | LabelsShape {c_labels.shape} | DataShapeN {c_data_n.shape} | DataShapeS {c_data_s.shape}")

            # Concatenate all the data and labels from all the companies
            self.data_n = torch_cat(all_data_n, dim = 0)
            self.data_s = torch_cat(all_data_s, dim = 0)
            self.labels = torch_cat(all_labels, dim = 0)

        # RNN
        else:
            
            # Create all the data sequences from each company
            all_data_sequences_n = []
            all_data_sequences_s = []
            all_labels = []
            all_dates = []

            # Data (Normalised and standardised versions)
            for i, (c_labels, c_dates, c_data_n, c_data_s) in enumerate(zip(self.labels, self.dates, self.data_n, self.data_s)):

                # The number of sequences of length "num_context_days" in this company's data
                num_sequences = c_labels.shape[0] - (num_context_days - 1) # E.g. if num_context_days = 10, 4530 --> (4530 - 10 - 1) = 

                # Trim labels (The same is done for self.data when converting to sequences)
                c_labels = c_labels[:num_sequences] # labels.shape = (Correct predictions for all sequences in self.data)

                # Trim dates
                c_dates = c_dates[:num_sequences]

                # Let num_context_days = 10, batch_size = 32
                # Single batch should be [10 x [32 * num_features] ]
                # 32 x [ClosingP, OpeningP, Volume, etc..] 10 days ago
                # The next batch for the recurrence will be the day after that day
                # 32 x [ClosingP, OpeningP, Volume, etc..] 9 days ago
                # Repeats until all 10 days have been passed in (for a single batch)
                # c_data.shape = (number of 10 consecutive days sequences, 10 consecutive days of examples, number of features in each day)
                c_data_n = torch_stack([c_data_n[i:i + num_context_days] for i in range(0, num_sequences)], dim = 0)
                c_data_s = torch_stack([c_data_s[i:i + num_context_days] for i in range(0, num_sequences)], dim = 0)

                # Add the data sequences and labels for this company to the lists
                all_data_sequences_n.append(c_data_n)
                all_data_sequences_s.append(c_data_s)

                # Add the labels for this company to the lists
                all_labels.append(c_labels)

                # Add the dates for this company to the list
                all_dates.extend(c_dates) # Extend so that it is a single list

                print(f"Company {i} | LabelsShape {c_labels.shape} | DataShapeN {c_data_n.shape} | DataShapeS {c_data_s.shape}")

            # Concatenate all the data sequences and labels from all the companies
            self.data_n = torch_cat(all_data_sequences_n, dim = 0)
            self.data_s = torch_cat(all_data_sequences_s)
            self.labels = torch_cat(all_labels, dim = 0)

        # Sorting / Shuffling data sequences
        if shuffle_data_sequences == True:
            self.shuffle_data_sequences() # Random shuffle
        else:
            self.sort_data_sequences(dates = all_dates) # Chronological order
        
        # Remove dates (no longer required)
        del self.dates
    
        print(f"DataShapeN: {self.data_n.shape} | DataShapeS: {self.data_s.shape} | LabelsShape: {self.labels.shape}")

    def separate_data_sequences(self):
        # Separates data sequences into training and test sets 
        # - The training set will be used for folds during training
        # - The test set will be used as a final evaluation of then model after cross-validation

        train_end_idx = int(self.labels.shape[0] * 0.8)
        
        self.TRAIN_SN = (self.data_n[0:train_end_idx], self.labels[0:train_end_idx])
        self.TRAIN_SS = (self.data_s[0:train_end_idx], self.labels[0:train_end_idx])
        self.TEST_SN = (self.data_n[train_end_idx:], self.labels[train_end_idx:])
        self.TEST_SS = (self.data_s[train_end_idx:], self.labels[train_end_idx:])
        
        print(f"TRAIN SET | Inputs: {self.TRAIN_SS[0].shape} | Labels: {self.TRAIN_SS[1].shape}")
        print(f"TEST SET | Inputs: {self.TEST_SS[0].shape} | Labels: {self.TEST_SS[1].shape}")
    
    def shuffle_data_sequences(self):
        # Shuffling the data sequences

        permutation_indices = torch_randperm(self.labels.shape[0], device = self.device, generator = self.generator) # Generate random permutation of indices
        permutation_indices = permutation_indices.to(device = "cpu") # Move to CPU as self.data is on the CPU

        # prev_data = self.data.clone()
        # prev_labels = self.labels.clone()
        # Assign indices to data and labels
        self.data_n = self.data_n[permutation_indices]
        self.data_s = self.data_s[permutation_indices]
        self.labels = self.labels[permutation_indices]
        
        # print(torch_equal(self.data, prev_data[permutation_indices]))
        # print(torch_equal(self.labels, prev_labels[permutation_indices])) 
    
    def sort_data_sequences(self, dates):
        
        # Convert the list of pandas timestamps into unix timestamps (Number of seconds that have elapsed since January 1, 1970 (UTC))
        # Notes: 
        # - Converted to unix timestamps so that we can perform torch.argsort()
        # - self.data and self.labels at this stage will be the companies data placed one after another, so the dates must be sorted into chronological order
        # - descending = True because the timestamps are seconds elapsed so larger numbers = further back in time
        unix_timestamps = torch_tensor([pd_to_datetime(time_stamp).timestamp() for time_stamp in dates])
        sort_indices = torch_argsort(unix_timestamps, descending = True) # Returns the indices which will sort self.data and self.labels in chronological order 

        #print("SORTED: DATALABELSDATES", self.data_n.shape, self.labels.shape, sort_indices.shape)
        
        # Sort data and labels
        self.data_n = self.data_n[sort_indices]
        self.data_s = self.data_s[sort_indices]
        self.labels = self.labels[sort_indices]

    def create_sets(self, num_context_days, shuffle_data_sequences):
        # Convert self.data_n, self.data_s, self.labels into data sequences 
        self.create_data_sequences(num_context_days = num_context_days, shuffle_data_sequences = shuffle_data_sequences)

        # Separate the data sequences into to two sets (Training and test)
        self.separate_data_sequences()

    def create_folds(self, num_folds, N_OR_S = "N"):
        # Creates folds out of the training set
        
        # TRAIN_SN or TRAIN_SS
        training_set = getattr(self, f"TRAIN_S{N_OR_S}")

        # Divide the data and labels into k folds
        # Note: self.d_folds_(n/s) and self.l_folds(n/s) will be a tuple of k folds, with each fold being a tensor
        setattr(self, f"d_folds_{N_OR_S.lower()}", torch_chunk(input = training_set[0], chunks = num_folds, dim = 0)) # self.d_folds_(n/s)
        setattr(self, f"l_folds_{N_OR_S.lower()}", torch_chunk(input = training_set[1], chunks = num_folds, dim = 0)) # self.l_folds_(n/s)
 
    def retrieve_k_folds(self, window_size, N_OR_S = "N"):
        
        # Implementation of walk-forward / expanding window cross validation:
        # = Selects the fold at the end of the window as the validation set and the remaining folds for training (k will be zero indexed)
        # - Models will always be trained on past data with the validation set being new unseen data.

        # Retrieve only the first "window_size" folds
        D_FOLDS = getattr(self, f"d_folds_{N_OR_S.lower()}")[:window_size]
        L_FOLDS = getattr(self, f"l_folds_{N_OR_S.lower()}")[:window_size]
        
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
            print("Loading labeled dataset")
            LABELED_DATA = pd_read_csv("sentiment_data/progress/labeled_tweets.csv")

        # Not created yet
        else:
            print("Creating labeled dataset")
            # Data containing the tweet id, writer, post date, tweet, number of comments, likes and retweets
            DATA = pd_read_csv("sentiment_data/Tweet.csv")

            # The IDs to all the tweets and what company they were referring to
            TWEET_COMPANY = pd_read_csv("sentiment_data/Company_Tweet.csv")

            # Remove any tweets if it was posted on dates that do not appear inside the historical dataset
            """ 
            - pd_date_time(__, unit = "s") to convert unix timestamps to dates
            - .dt.date to round the dates from e.g. 2015-01-01 00:01:36 to 2015-01-01 00:00:00
            """
            print(DATA)
            DATA["post_date"] = pd_to_datetime(DATA["post_date"], unit = "s").dt.date # Convert unix timestamps to dates in the data column
            DATA = DATA[DATA["post_date"].isin(self.dates)]
            print(DATA)

            # Merge the company tickers with each tweet according to tweet id
            MERGED = DATA.merge(TWEET_COMPANY, on = "tweet_id", how = "left")

            print(MERGED[MERGED["body"] == ""])

            # print(DATA[DATA.isna().any(axis = 1)])
            # print(TWEET_COMPANY[TWEET_COMPANY.isna().any(axis = 1)])
            # print(MERGED[MERGED.isna().any(axis = 1)])

            print(MERGED[MERGED["body"].isna()])

            # Label the merged dataset with sentiment values
            LABELED_DATA = self.label_dataset(dataset = MERGED)

        print(LABELED_DATA)
        
        #print(pd_read_csv("sentiment_data/labeled_tweets.csv"))

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
        #     print("!", tweet, f"Ticker: {ticker}")
        
        # Apply cleaning to all text
        dataset["body"] = dataset["body"].apply(lambda x: cleanse_text(x))

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
        prompt_dataset["sentiments"] = all_sentiments

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
        labeled_tweets.to_csv("sentiment_data/progress/labeled_tweets.csv", index = True)

        return labeled_tweets

