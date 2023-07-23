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

class DataHandler:

    def __init__(self, device, generator):

        self.device = device # CUDA or CPU
        self.generator = generator # Reproducibility

        # self.data - Holds the inputs for each example
        # self.labels - Holds the corresponding targets for each example
        # self.TRAIN_S(N/S) - Training set
        # self.TEST_S(N/S) - Test set
        # self.TRAIN_FOLDS - Training folds
        # self.VAL_FOLDS - Validation folds
        
        # self.n_features - Number of inputs that will be passed into a model (i.e. the number of columns/features in the pandas dataframe)

    def retrieve_data(self, tickers, start_date, end_date, interval):
        
        self.data_n = [] # Normalised data
        self.data_s = [] # Standardised data
        self.labels = []
        
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
            # Notes:
            # - Created 2 because some models may perform better on standardised data than normalised data and vice versa
            # - Min-max normalisation preserves relative relationships between data points but eliminates differences in magnitude
            # - Standardisation brings data features onto a similar scale to be comparable (Helps remove the influence of the mean and scale of data where distribution of data is not Gaussian or contains outliers)
            cols_to_alter = ["open", "high", "low", "close", "adjclose", "volume"]
            S_DATA = DATA.copy()
            S_DATA[cols_to_alter] = self.standardise_columns(dataframe = S_DATA, cols_to_standard = cols_to_alter)
            DATA[cols_to_alter] = self.normalise_columns(dataframe = DATA, cols_to_norm = cols_to_alter)
            
            # Add this companies data to the list
            self.data_n.append(self.dataframe_to_ptt(pandas_dataframe = DATA, desired_dtype = torch_float_32))
            self.data_s.append(self.dataframe_to_ptt(pandas_dataframe = S_DATA, desired_dtype = torch_float_32))

            print(f"Ticker: {ticker} | DataShape: {self.data_n[-1].shape} | LabelsShape: {self.labels[-1].shape}")
        
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

            cr_column_name = f"CloseRation_{p}"
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

    def dataframe_to_ptt(self, pandas_dataframe, desired_dtype = torch_float_32):
        
        # Converts a pandas dataframe to a PyTorch tensor (With the default dtype being torch.float32)
        # 1. pandas_dataframe.values extracts the data into numpy arrays (dtype = float64) [Maintains shape]
        # 2. torch_tensor converts np array to PyTorch tensor

        return torch_tensor(pandas_dataframe.values, dtype = desired_dtype)

    def generate_batch(self, batch_size, dataset, num_context_days):
        
        # Find the inputs and labels in the dataset and find the number of examples in this set
        inputs, labels = dataset
        num_examples = labels.shape[0]
        
        # Generate indexes which correspond to each example in the labels and inputs of this dataset (perform using CUDA if possible)
        u_distrib = torch_ones(num_examples, device = self.device) / num_examples # Uniform distribution
        example_idxs = torch_multinomial(input = u_distrib, num_samples = batch_size, replacement = True, generator = self.generator)

        # Move indices to the "cpu" so that we can index the dataset, which is currently stored on the CPU
        example_idxs = example_idxs.to(device = "cpu")

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

    def create_data_sequences(self, num_context_days):
        # Converts self.data_n, self.data_s, self.labels into data sequences

        # MLP 
        if num_context_days == 1:
            
            # Combine all companies data together
            # data.shape = [number of single day examples, number of features for each day]
            # labels.shape = [number of single day examples, correct prediction for stock trend for the following day]
            
            all_data_n = []
            all_data_s = []
            all_labels = []

            for i, (c_labels, c_data_n, c_data_s) in enumerate(zip(self.labels, self.data_n, self.data_s)):
                # Add the data and labels for this company to the lists
                all_data_n.append(c_data_n)
                all_data_s.append(c_data_s)
                all_labels.append(c_labels)
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

            # Data (Normalised and standardised versions)
            for i, (c_labels, c_data_n, c_data_s) in enumerate(zip(self.labels, self.data_n, self.data_s)):

                # The number of sequences of length "num_context_days" in this company's data
                num_sequences = c_labels.shape[0] - (num_context_days - 1) # E.g. if num_context_days = 10, 4530 --> (4530 - 10 - 1) = 

                # Trim labels (The same is done for self.data when converting to sequences)
                c_labels = c_labels[:num_sequences] # labels.shape = (Correct predictions for all sequences in self.data)

                # Add the labels for this company to the lists
                all_labels.append(c_labels)

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
                
                print(f"Company {i} | LabelsShape {c_labels.shape} | DataShapeN {c_data_n.shape} | DataShapeS {c_data_s.shape}")

            # Concatenate all the data sequences and labels from all the companies
            self.data_n = torch_cat(all_data_sequences_n, dim = 0)
            self.data_s = torch_cat(all_data_sequences_s)
            self.labels = torch_cat(all_labels, dim = 0)
        
        print(f"DataShapeN: {self.data_n.shape} | DataShapeS: {self.data_s.shape} | LabelsShape: {self.labels.shape}")

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
    
    def separate_data_sequences(self):
        # Separates data sequences into training and test sets 
        # - The training set will be used for folds during training
        # - The test set will be used as a final evaluation of then model after cross-validation

        train_end_idx = int(self.labels.shape[0] * 0.8)
        
        self.TRAIN_SN = (self.data_n[0:train_end_idx], self.labels[0:train_end_idx])
        self.TRAIN_SS = (self.data_s[0:train_end_idx], self.labels[0:train_end_idx])
        self.TEST_SN = (self.data_n[train_end_idx:], self.labels[train_end_idx:])
        self.TEST_SS = (self.data_s[train_end_idx:], self.labels[train_end_idx:])

        print(train_end_idx)
        print(f"TRAIN SET | Inputs: {self.TRAIN_SS[0].shape} | Labels: {self.TRAIN_SS[1].shape}")
        print(f"TEST SET | Inputs: {self.TEST_SS[0].shape} | Labels: {self.TEST_SS[1].shape}")
    
    def create_sets(self, num_context_days):
        # Convert self.data_n, self.data_s, self.labels into data sequences 
        self.create_data_sequences(num_context_days = num_context_days)

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
 
    def retrieve_k_folds(self, k, N_OR_S = "N"):
        
        # Select the kth fold as the validation set and the remaining folds for training (k will be zero indexed)
        # Note: Will overwrite the previous self.TRAIN_FOLDS and self.VAL_FOLDS after selecting a new fold

        D_FOLDS = getattr(self, f"d_folds_{N_OR_S.lower()}")
        L_FOLDS = getattr(self, f"l_folds_{N_OR_S.lower()}")

        print(f"TRAIN FOLDS | Inputs: {torch_cat([D_FOLDS[i] for i in range(len(D_FOLDS)) if i != k]).shape} | Labels: {torch_cat([L_FOLDS[i] for i in range(len(D_FOLDS)) if i != k]).shape}")
        print(f"VAL FOLD | Inputs: {D_FOLDS[k].shape} | Labels: {L_FOLDS[k].shape}")

        # Training folds and validation fold
        # Note: Each tuple = (data folds, corresponding label folds)
        # Return the training folds  and validation folds
        return (torch_cat([D_FOLDS[i] for i in range(len(D_FOLDS)) if i != k]), torch_cat([L_FOLDS[i] for i in range(len(D_FOLDS)) if i != k])), (D_FOLDS[k], L_FOLDS[k])