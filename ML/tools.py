from torch import no_grad as torch_no_grad
from torch import max as torch_max
from torch import sum as torch_sum
from torch.nn import functional as F
from torch import logical_and as torch_logical_and
    
def count_correct_preds(predictions, targets):

    # Selects the highest probability assigned between the two outputs (i.e. the highest probability of the stock price going up or down)
    _, output = torch_max(predictions, dim = 1) 
    
    # Return the number of correct predictions
    return torch_sum(output == targets).item()

def find_P_A_R(predictions, targets):
    # Finds the precision, accuracy and recall for a given batch

    # Selects the highest probability assigned between the two outputs (i.e. the highest probability of the stock price going up or down)
    _, outputs = torch_max(predictions, dim = 1) 

    # Number of true positives (prediction == 1 when target == 1)
    true_positives = torch_logical_and(outputs == 1, targets == 1).sum().item()

    # Number of false positives (prediction == 1 when target == 0) 
    false_positives = torch_logical_and(outputs == 1, targets == 0).sum().item()
    
    # Number of false negatives (prediction == 0 when target == 0)
    true_negatives = torch_logical_and(outputs == 0, targets == 0).sum().item()

    # Number of false negatives (prediction == 0 when target == 1)
    false_negatives = torch_logical_and(outputs == 0, targets == 1).sum().item()

    # Accuracy = Number of correct predictions / Total number of predictions
    # Precision = Number of true positives / (Number of true positives + Number of false positives) [Proportion of true positive predictions that the stock went up out of all predictions out of all the predictions that the model made that the stock went up]
    # Recall = Number of true positives / (Number of true positives + Number of false negatives) [Proportion of true positive predictions that the stock went up out of all the times it actually went up in the dataset]
    # F1 score = 2 * ((precision * recall) / (precision + recall))

    # Notes: 
    # - Higher F1 score is better (Ranges from 0 to 1)
    # - Recall measures the ability of the model to identify positive instances (stock trend going up) out of all the positive instances that occurred
    # - Precision measures the accuracy of actual positive predictions amongst all the instances predicted as positive by the model
    # - If precision or recall is 0, F1 score will also be 0
    accuracy = ((true_positives + true_negatives) / predictions.shape[0]) * 100
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * ((precision * recall) / (precision + recall)) if precision != 0 or recall != 0 else 0

    # print("O",outputs)
    # print("T",targets)
    # print(true_positives, false_positives, true_negatives, false_negatives)
    # print(accuracy, precision, recall, f1_score)

    return accuracy, precision, recall, f1_score

def get_predictions(input_data, model):
    """
    input_data shape should be: [batch_size, num_context days, num_features]
    """
    with torch_no_grad():
        # Find probability distribution (the predictions)
        return F.softmax(model(input_data), dim = 1)

def get_model_prediction(model_number_load, tickers):

    # Import dependencies
    from torch import manual_seed as torch_manual_seed
    from torch.cuda import manual_seed_all as torch_cuda_manual_seed_all
    from torch import Generator as torch_Generator
    from data_handler import DataHandler
    from model_manager import ModelManager
    from os import getenv as os_getenv
    from os.path import exists as os_path_exists
    from os import mkdir as os_mkdir
    from requests import get as requests_get
    from pandas import read_csv as pd_read_csv
    from time import sleep as time_sleep
    from pandas import DataFrame as pd_DataFrame
    from torch import float32 as torch_float32
    from torch import concat as torch_concat
    from torch import zeros as torch_zeros
    from torch import stack as torch_stack
    from torch import argmax as torch_argmax

    DEVICE = "cpu" # Faster inference times on CPU
    M_SEED = 2004
    torch_manual_seed(M_SEED)
    torch_cuda_manual_seed_all(M_SEED)
    G = torch_Generator(device = DEVICE)
    G.manual_seed(M_SEED)
    DH = DataHandler(device = DEVICE, generator = G)
    model_manager = ModelManager(device = DEVICE, DH_reference = DH, TDH_reference = None)

    """
    model_number_load = Number of the model to load, leave empty to create a new model
    .initiate_model = Returns model, optimiser and hyperparamaters used to train the model

    If using for training / testing on the testing set:
        - Will use DH.retrieve_data before instantiating the model if creating a new model
        - Will use DH.retrieve_data after instantiating the model if loading an existing model
    """
    model, _, hyperparameters, _, _= model_manager.initiate_model(
                                                                model_number_load = model_number_load, 
                                                                manual_hyperparams = None, 
                                                                inference = True
                                                                )

    # print(f"Hyperparameters used: {hyperparameters}")
    # print(f"Model architecture: {model.__class__.__name__} | Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Retrieve and process data

    api_key = os_getenv("alphavantage_apikey")
    actual_column_names = ["open", "high", "low", "close", "volume"]
    info_list = []

    if os_path_exists("historical_data") == False:
        os_mkdir("historical_data")

    for ticker in tickers:
        path_to_data = f"historical_data/{ticker}.csv"
        if os_path_exists(path_to_data):
            # print(f"Loading {ticker} data from disk")
            # Read the dataframe from the path
            # - parse_dates to parse the values in the first column (i.e. the dates) as datetime objects
            # - index_col to specify which column should be used as the index of the dataframe
            DATAFRAME = pd_read_csv(path_to_data, parse_dates = True, index_col = 0)
            
        else:
            # print("Calling API")
            # Get data on single ticker
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'
            r = requests_get(url)
            api_data = r.json()

            time_sleep(16)

            # Received an error, move to the next ticker
            if api_data.get("Error Message") != None: 
                continue
            
            # print(api_data)
            # print(api_data.keys())
            # print(api_data["Time Series (Daily)"].keys())

            # Create dataframe
            """ Notes:
            - .T = Transpose to get ["open", "high", "low", "close", "volume"] as the columns
            - .astype(float) to convert the dtypes from "object" to "float64"
            - .sort_values(by = DATAFRAME.index, ascending = True) to sort the rows by their dates (as)
            """
            DATAFRAME = pd_DataFrame(api_data["Time Series (Daily)"]).T.astype(float)

            # Save dataframe with the dates
            DATAFRAME.to_csv(path_to_data, date_format = "%Y-%m-%d", index = True)

        # Sort dataframe based on the dates (Originally from latest dates to earliest dates)
        DATAFRAME.sort_index(ascending = True, inplace = True)

        # Rename columns
        rename_columns = {prev_name: new_name for prev_name, new_name in zip(DATAFRAME.columns, actual_column_names)}
        DATAFRAME.rename(columns = rename_columns, inplace = True)
        # print("Renamed", DATAFRAME)

        # Modify data
        # Note: Only take the last hyperparameters["num_context_days"] days to create a single data sequence to predict the closing price today
        DATAFRAME = DH.modify_data(DATAFRAME, dated_sentiments = None, include_date_before_prediction_date = True, hyperparameters = hyperparameters)[-hyperparameters["num_context_days"]:]
        # print("Modified", DATAFRAME)

        # Remove labels
        DATAFRAME.drop("Target", axis = 1, inplace = True)

        # Standardise / Normalise
        if hyperparameters["transform_after"] == True: # Transformation based on "N_OR_S" and "transform_data"
            transformation = DH.standardise_data if hyperparameters["N_OR_S"] == "S" else DH.normalise_data
        else:
            transformation = DH.standardise_columns if hyperparameters["N_OR_S"] == "S" else DH.normalise_columns
        
        if hyperparameters["transform_after"] == False:
            # Alter the columns first
            DATAFRAME[hyperparameters["cols_to_alter"]] = DH.normalise_columns(dataframe = DATAFRAME, cols_to_norm = hyperparameters["cols_to_alter"])

            # Convert to dataframe afterwards
            DH.data = DH.dataframe_to_ptt(pandas_dataframe = DATAFRAME, desired_dtype = torch_float32)

        else:
            """Note
            - Convert to tensors first, then transform the desired columns inside each tensor
            - Transformation is usually over multiple companies so DH.data is usually a Python list, containing the dataframes for each company
            """
            DH.data = [DH.dataframe_to_ptt(pandas_dataframe = DATAFRAME, desired_dtype = torch_float32)]

            # Transform the data in the desired columns
            col_indexes = [DATAFRAME.columns.get_loc(column_name) for column_name in hyperparameters["cols_to_alter"]] # Find indexes of all the columns we want to alter
            transformation(combined_data = DH.data, col_indexes = col_indexes, params_from_training = hyperparameters["train_data_params"])

            # Only select the data for this company (DH.data[0])
            data_sequence = DH.data[0] 

        # Create single data sequence
        if model.__class__.__name__ == "MLP": 
            # Convert from [batch_size, num_features] --> [num_features] (2D tensor to 1D tensor)
            data_sequence = data_sequence.view(data_sequence.shape[1])
        elif model.__class__.__name__ == "RNN": 
            # Convert from [num_context_days, num_features] --> [batch_size, num_context_days, num_features] --> [num_context_days, batch_size, num_features]
            data_sequence = data_sequence.view(1, data_sequence.shape[0], -1).transpose(dim0 = 0, dim1 = 1)
        
        # Add to data sequences dict
        # print(DATAFRAME.index)
        info_list.append({"data_sequence": data_sequence, "date_before_prediction_date": DATAFRAME.index[-1], "ticker": ticker})

    # hyperparameters["batch_size"] = 32
    num_tickers = len(info_list) # Can be different to len(tickers) if the data was not retrieved successfully
    # print(num_tickers)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create batches
    """
    batches.shape = [Number of batches, num_context_days, batch_size, num_features]
    batch.shape = [num_context_days, batch_size, num_features]
    """
    hyperparameters["batch_size"] = 3
    all_data_sequences = [company_dict["data_sequence"] for company_dict in info_list]

    if model.__class__.__name__ == "RNN":
        batches = [torch_concat(all_data_sequences[i:i + hyperparameters["batch_size"]], dim = 1) for i in range(0, num_tickers, hyperparameters["batch_size"])]

        # Padding final batch for batch prompting
        # - Pads to the right of batches that aren't the same size as the batch size used to train the model with
        if batches[-1].shape[1] != hyperparameters["batch_size"]:
            right_padding = torch_zeros(hyperparameters["num_context_days"], hyperparameters["batch_size"] - batches[-1].shape[1], batches[-1].shape[2])
            batches[-1] = torch_concat(tensors = [batches[-1], right_padding], dim = 1) # Concatenate on the batch dimension
            # print(batches[-1].shape)
            # print("R", right_padding.shape)
            # print("Padded", batches[-1].shape)

    elif model.__class__.__name__ == "MLP":
            batches = [torch_stack(all_data_sequences[i:i + hyperparameters["batch_size"]], dim = 0) for i in range(0, num_tickers, hyperparameters["batch_size"])]

            # Padding final batch for batch prompting
            if batches[-1].shape[0] != hyperparameters["batch_size"]:
                right_padding = torch_zeros(hyperparameters["batch_size"] - batches[-1].shape[0], batches[-1].shape[1])
                batches[-1] = torch_concat(tensors = [batches[-1], right_padding], dim = 0) # Concatenate on the batch dimension
                # print(batches[-1].shape)
                # print("R", right_padding.shape)
                # print("Padded", batches[-1].shape)
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Generate predictions
    # - Concatenate all of the predictions together, ignoring any predictions for padding sequences
    predictions = torch_concat([model(inputs = batch.to(device = DEVICE)) for batch in batches], dim = 0)[:num_tickers]

    # Add additional info into the company dicts
    for i in range(len(info_list)):
        info_list[i]["prediction"] = predictions[i]
        info_list[i]["ModelAnswer"] = torch_argmax(predictions[i], dim = 0).item()
        del data_sequence
        # print({key: info_list[i][key] for key in info_list[i].keys() if key != "data_sequence"})

    return info_list

def get_model_prediction_2(model_number_load, tickers):

    # Import dependencies
    import torch
    from data_handler import DataHandler
    from model_manager import ModelManager
    import requests
    import pandas as pd
    from os import getenv as os_get_env
    from time import sleep as time_sleep
    from os.path import exists as os_path_exists
    from os import mkdir as os_mkdir
    from torch import float32 as torch_float32

    DEVICE = "cpu" # Faster inference times on CPU
    M_SEED = 2004
    torch.manual_seed(M_SEED)
    torch.cuda.manual_seed_all(M_SEED)
    G = torch.Generator(device = DEVICE)
    G.manual_seed(M_SEED)
    DH = DataHandler(device = DEVICE, generator = G)
    model_manager = ModelManager(device = DEVICE, DH_reference = DH, TDH_reference = None)

    """
    model_number_load = Number of the model to load, leave empty to create a new model
    .initiate_model = Returns model, optimiser and hyperparamaters used to train the model

    If using for training / testing on the testing set:
        - Will use DH.retrieve_data before instantiating the model if creating a new model
        - Will use DH.retrieve_data after instantiating the model if loading an existing model
    """
    model, _, hyperparameters, _, _= model_manager.initiate_model(
                                                                model_number_load = model_number_load, 
                                                                manual_hyperparams = None, 
                                                                inference = True
                                                                )

    # print(f"Hyperparameters used: {hyperparameters}")
    # print(f"Model architecture: {model.__class__.__name__} | Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Retrieve and process data

    api_key = os_get_env("alphavantage_apikey")
    actual_column_names = ["open", "high", "low", "close", "volume"]
    info_list = []

    if os_path_exists("historical_data") == False:
        os_mkdir("historical_data")

    for ticker in tickers:
        path_to_data = f"historical_data/{ticker}.csv"
        if os_path_exists(path_to_data):
            # print(f"Loading {ticker} data from disk")
            # Read the dataframe from the path
            # - parse_dates to parse the values in the first column (i.e. the dates) as datetime objects
            # - index_col to specify which column should be used as the index of the dataframe
            DATAFRAME = pd.read_csv(path_to_data, parse_dates = True, index_col = 0)
            
        else:
            # print("Calling API")
            # Get data on single ticker
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'
            r = requests.get(url)
            api_data = r.json()

            time_sleep(16)

            # Received an error, move to the next ticker
            if api_data.get("Error Message") != None: 
                continue
            
            # print(api_data)
            # print(api_data.keys())
            # print(api_data["Time Series (Daily)"].keys())

            # Create dataframe
            """ Notes:
            - .T = Transpose to get ["open", "high", "low", "close", "volume"] as the columns
            - .astype(float) to convert the dtypes from "object" to "float64"
            - .sort_values(by = DATAFRAME.index, ascending = True) to sort the rows by their dates (as)
            """
            DATAFRAME = pd.DataFrame(api_data["Time Series (Daily)"]).T.astype(float)

            # Save dataframe with the dates
            DATAFRAME.to_csv(path_to_data, date_format = "%Y-%m-%d", index = True)

        # Sort dataframe based on the dates (Originally from latest dates to earliest dates)
        DATAFRAME.sort_index(ascending = True, inplace = True)

        # Rename columns
        rename_columns = {prev_name: new_name for prev_name, new_name in zip(DATAFRAME.columns, actual_column_names)}
        DATAFRAME.rename(columns = rename_columns, inplace = True)
        # print("Renamed", DATAFRAME)

        # Modify data
        # Note: Only take the last hyperparameters["num_context_days"] days to create a single data sequence to predict the closing price today
        DATAFRAME = DH.modify_data(DATAFRAME, dated_sentiments = None, include_date_before_prediction_date = True, hyperparameters = hyperparameters)[-hyperparameters["num_context_days"]:]
        # print("Modified", DATAFRAME)

        # Remove labels
        DATAFRAME.drop("Target", axis = 1, inplace = True)

        # Standardise / Normalise
        if hyperparameters["transform_after"] == True: # Transformation based on "N_OR_S" and "transform_data"
            transformation = DH.standardise_data if hyperparameters["N_OR_S"] == "S" else DH.normalise_data
        else:
            transformation = DH.standardise_columns if hyperparameters["N_OR_S"] == "S" else DH.normalise_columns
        
        if hyperparameters["transform_after"] == False:
            # Alter the columns first
            DATAFRAME[hyperparameters["cols_to_alter"]] = DH.normalise_columns(dataframe = DATAFRAME, cols_to_norm = hyperparameters["cols_to_alter"])

            # Convert to dataframe afterwards
            DH.data = DH.dataframe_to_ptt(pandas_dataframe = DATAFRAME, desired_dtype = torch_float32)

        else:
            """Note
            - Convert to tensors first, then transform the desired columns inside each tensor
            - Transformation is usually over multiple companies so DH.data is usually a Python list, containing the dataframes for each company
            """
            DH.data = [DH.dataframe_to_ptt(pandas_dataframe = DATAFRAME, desired_dtype = torch_float32)]

            # Transform the data in the desired columns
            col_indexes = [DATAFRAME.columns.get_loc(column_name) for column_name in hyperparameters["cols_to_alter"]] # Find indexes of all the columns we want to alter
            transformation(combined_data = DH.data, col_indexes = col_indexes, params_from_training = hyperparameters["train_data_params"])

            # Only select the data for this company (DH.data[0])
            data_sequence = DH.data[0] 

        # Create single data sequence
        if model.__class__.__name__ == "MLP": 
            # Convert from [batch_size, num_features] --> [num_features] (2D tensor to 1D tensor)
            data_sequence = data_sequence.view(data_sequence.shape[1])
        elif model.__class__.__name__ == "RNN": 
            # Convert from [num_context_days, num_features] --> [batch_size, num_context_days, num_features] --> [num_context_days, batch_size, num_features]
            data_sequence = data_sequence.view(1, data_sequence.shape[0], -1).transpose(dim0 = 0, dim1 = 1)
        
        # Add to data sequences dict
        # print(DATAFRAME.index)
        info_list.append({"data_sequence": data_sequence, "date_before_prediction_date": DATAFRAME.index[-1], "ticker": ticker})

    # hyperparameters["batch_size"] = 32
    num_tickers = len(info_list) # Can be different to len(tickers) if the data was not retrieved successfully
    # print(num_tickers)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create batches
    """
    batches.shape = [Number of batches, num_context_days, batch_size, num_features]
    batch.shape = [num_context_days, batch_size, num_features]
    """
    hyperparameters["batch_size"] = 3
    all_data_sequences = [company_dict["data_sequence"] for company_dict in info_list]

    if model.__class__.__name__ == "RNN":
        batches = [torch.concat(all_data_sequences[i:i + hyperparameters["batch_size"]], dim = 1) for i in range(0, num_tickers, hyperparameters["batch_size"])]

        # Padding final batch for batch prompting
        # - Pads to the right of batches that aren't the same size as the batch size used to train the model with
        if batches[-1].shape[1] != hyperparameters["batch_size"]:
            right_padding = torch.zeros(hyperparameters["num_context_days"], hyperparameters["batch_size"] - batches[-1].shape[1], batches[-1].shape[2])
            batches[-1] = torch.concat(tensors = [batches[-1], right_padding], dim = 1) # Concatenate on the batch dimension
            # print(batches[-1].shape)
            # print("R", right_padding.shape)
            # print("Padded", batches[-1].shape)

    elif model.__class__.__name__ == "MLP":
            batches = [torch.stack(all_data_sequences[i:i + hyperparameters["batch_size"]], dim = 0) for i in range(0, num_tickers, hyperparameters["batch_size"])]

            # Padding final batch for batch prompting
            if batches[-1].shape[0] != hyperparameters["batch_size"]:
                right_padding = torch.zeros(hyperparameters["batch_size"] - batches[-1].shape[0], batches[-1].shape[1])
                batches[-1] = torch.concat(tensors = [batches[-1], right_padding], dim = 0) # Concatenate on the batch dimension
                # print(batches[-1].shape)
                # print("R", right_padding.shape)
                # print("Padded", batches[-1].shape)
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Generate predictions
    # - Concatenate all of the predictions together, ignoring any predictions for padding sequences
    predictions = torch.concat([model(inputs = batch.to(device = DEVICE)) for batch in batches], dim = 0)[:num_tickers]

    # Add additional info into the company dicts
    for i in range(len(info_list)):
        info_list[i]["prediction"] = predictions[i]
        info_list[i]["ModelAnswer"] = torch.argmax(predictions[i], dim = 0).item()
        del data_sequence
        # print({key: info_list[i][key] for key in info_list[i].keys() if key != "data_sequence"})

    return info_list