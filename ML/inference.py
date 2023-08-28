from time import time as get_time

def get_model_prediction(ticker, date_to_predict):

    # Importing dependencies + model set-up
    setup_time_1 = get_time()
    import torch
    from data_handler import DataHandler
    from model_manager import ModelManager
    from datetime import datetime, timedelta
    from yahoo_fin.stock_info import get_data

    DEVICE = "cpu" # Faster inference times on CPU
    M_SEED = 2004
    torch.manual_seed(M_SEED)
    torch.cuda.manual_seed_all(M_SEED)
    G = torch.Generator(device = DEVICE)
    G.manual_seed(M_SEED)
    setup_time_2 = get_time()

    model_load_time_1 = get_time()
    DH = DataHandler(device = DEVICE, generator = G)
    model_manager = ModelManager(device = DEVICE, DH_reference = DH, TDH_reference = None)
    model_number_load = 23
    model, _, hyperparameters, _, _ = model_manager.initiate_model(
                                                                    model_number_load = model_number_load, 
                                                                    manual_hyperparams = None, 
                                                                    inference = True
                                                                    )

    model_load_time_2 = get_time()
    # print(f"Hyperparameters used: {hyperparameters}")
    # print(f"Model architecture: {model.__class__.__name__} | Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Creating data sequence

    data_load_time_1 = get_time()
    
    # Find starting date
    num_days_prev = (hyperparameters["num_context_days"] + hyperparameters["rolling_periods"][-1]) * 2
    end_date_as_dt = datetime.strptime(date_to_predict, "%d/%m/%Y")
    start_date_as_dt = end_date_as_dt - timedelta(days = num_days_prev)

    # Convert to MM/DD/YYYY
    start_date = start_date_as_dt.strftime("%m/%d/%Y")
    end_date = end_date_as_dt.strftime("%m/%d/%Y") 

    # print(hyperparameters["rolling_periods"][-1])
    # print(hyperparameters["num_context_days"])
    # print(num_days_prev)
    # print(start_date, end_date)
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Creating the data sequence

    # Retrieve data on the ticker
    DATA = get_data(
                    ticker = ticker, 
                    start_date = start_date, 
                    end_date = end_date, 
                    index_as_date = True, 
                    interval = "1d"
                    )

    # Modify data
    DATA = DH.modify_data(
                        D = DATA, 
                        dated_sentiments = None, 
                        include_date_before_prediction_date = True, 
                        hyperparameters = hyperparameters
                        )
    # Remove labels
    DATA.drop("Target", axis = 1, inplace = True)

    # Transform in the context of itself (the company)
    if hyperparameters["transform_after"] == False:
        # Standardise
        if hyperparameters["N_OR_S"] == "S":
            DATA = (DATA[hyperparameters["cols_to_alter"]] - DATA[hyperparameters["cols_to_alter"]].mean()) / DATA[hyperparameters["cols_to_alter"]].std()
        # Normalise
        else:
            DATA[hyperparameters["cols_to_alter"]] = (DATA[hyperparameters["cols_to_alter"]] - DATA[hyperparameters["cols_to_alter"]].min()) / (DATA[hyperparameters["cols_to_alter"]].max() - DATA[hyperparameters["cols_to_alter"]].min())
        DH.data = DH.dataframe_to_ptt(pandas_dataframe = DATA, desired_dtype = torch.float32)

    # Transform in the context of all companies (i.e. the companies used in training)
    else:
        col_indexes = [DATA.columns.get_loc(column_name) for column_name in hyperparameters["cols_to_alter"]]
        DH.data = DH.dataframe_to_ptt(pandas_dataframe = DATA, desired_dtype = torch.float32)

        # Standardise
        if hyperparameters["N_OR_S"] == "S":
            DH.data[:,col_indexes] -= hyperparameters["train_data_params"]["S"]["mean"]
            DH.data[:,col_indexes] /= hyperparameters["train_data_params"]["S"]["std"]
        # Normalise
        else:
            DH.data[:, col_indexes] -= hyperparameters["train_data_params"]["N"]["minimums"]
            DH.data[:, col_indexes] /= hyperparameters["train_data_params"]["N"]["differences"]
    
    # Create data sequence (Taking only "num_context_days" data points)
    DATA_SEQUENCE = DH.data[-hyperparameters["num_context_days"]:]
    print(DATA_SEQUENCE.shape)
    data_load_time_2 = get_time()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create batch


    # - batch.shape = [num_context_days, batch_size, num_features]
    batch_create_time_1 = get_time()

    if model.__class__.__name__ == "RNN":
        # Sequence shape should be [num_context_days, 1, num_features]
        right_padding = torch.zeros(hyperparameters["num_context_days"], hyperparameters["batch_size"] - 1, hyperparameters["n_features"])
        batch = torch.concat([DATA_SEQUENCE.view(hyperparameters["num_context_days"], 1, hyperparameters["n_features"]), right_padding], dim = 1)
        print(right_padding.shape)
        print(batch.shape)

    elif model.__class__.__name__ == "MLP":
        # Sequence shape should be [1, num_features]
        right_padding = torch.zeros(hyperparameters["batch_size"] - 1, hyperparameters["n_features"])
        batch = torch.concat([DATA_SEQUENCE.view(1, hyperparameters["n_features"]), right_padding], dim = 0)
        print(right_padding.shape)
        print(batch.shape)
    
    batch_create_time_2 = get_time()

    prediction_time_1 = get_time()
    # Generate prediction
    prediction = torch.nn.functional.softmax(model(inputs = batch.to(device = DEVICE)), dim = 1)[0]
    prediction_time_2 = get_time()

    print(prediction.shape)

    print("SetUpTime", setup_time_2 - setup_time_1)
    print("ModelLoadTime", model_load_time_2 - model_load_time_1)
    print("DataLoadTime", data_load_time_2 - data_load_time_1)
    print("BatchTime", batch_create_time_2 - batch_create_time_1)
    print("PredictionTime", prediction_time_2 - prediction_time_1)

    total_time = sum(
                    [
                    setup_time_2 - setup_time_1,
                    model_load_time_2 - model_load_time_1,
                    data_load_time_2 - data_load_time_1,
                    batch_create_time_2 - batch_create_time_1,
                    prediction_time_2 - prediction_time_1,
                    ]
                    )               
    print("Total time: ", total_time)
    
    return {"date_to_predict": date_to_predict, "ticker": ticker, "prediction": prediction, "model_answer": torch.argmax(prediction, dim = 0).item()}

return_dict = get_model_prediction(ticker = "meta" , date_to_predict = "28/08/2023")
print(return_dict)