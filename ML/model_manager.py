from os import listdir as os_listdir
from os import mkdir as os_mkdir
from os import rmdir as os_rmdir
from os.path import exists as os_path_exists
from torch import load as torch_load
from torch.optim import Adam as torch_optim_Adam
from torch.optim import SGD as torch_optim_SGD
from models import MLP, RNN, LSTM
from torch import Tensor as torch_Tensor

class ModelManager:

    def __init__(self, device, DH_reference, TDH_reference):

        self.device = device

        self.DH_REF = DH_reference
        self.TDH_REF = TDH_reference

    def initiate_model(self, model_number_load = None, manual_hyperparams = None, inference = False, feature_test_model_name = None):

        model_checkpoints_folder_exists = os_path_exists("model_checkpoints") 
        # Load existing model
        if model_number_load != None and model_checkpoints_folder_exists:
            # Either deployment model or feature test model (dependent on feature_test_model_name parameter)
            checkpoint_directory = f"model_checkpoints/{'feature_test_models/' if feature_test_model_name else 'deployment_models/'}{model_number_load}"
            self.clean_empty_directories()

            # print("Loading existing model")
            existing_checkpoint_path = f"{checkpoint_directory}/fold_{len(os_listdir(f'{checkpoint_directory}')) - 1}.pth"
            checkpoint = torch_load(existing_checkpoint_path) # Load the last checkpoint (Which would be the complete model)
            # print(existing_checkpoint_path)
            # print(checkpoint.keys())

            """ 
            - checkpoint is a dict consisting of 3 keys: 
                - "model" - Model + Optimiser state dicts and architecture used
                - "hyperparameters" - Hyperparameters used to train the model
                - "stats": Model statistics (e.g. loss, accuracy, f1 score, etc..)
            - Optimiser is not loaded if called for model inference (not on the test set)
            """
            hyperparameters = checkpoint["hyperparameters"] # Load hyperparameters of the saved model

            if checkpoint["model"]["architecture"] == "RNN":
                model = RNN(initial_in = hyperparameters["n_features"], final_out = 2, N_OR_S = hyperparameters["N_OR_S"], uses_single_sentiments = hyperparameters["uses_single_sentiments"])
                if inference == False:
                    optimiser = torch_optim_Adam(params = model.parameters(), lr = hyperparameters["learning_rate"])
            
            elif checkpoint["model"]["architecture"] == "MLP":
                model = MLP(initial_in = hyperparameters["n_features"], final_out = 2, N_OR_S = hyperparameters["N_OR_S"], uses_single_sentiments = hyperparameters["uses_single_sentiments"])
                if inference == False:
                    optimiser = torch_optim_SGD(params = model.parameters(), lr = hyperparameters["learning_rate"])
            
            elif checkpoint["model"]["architecture"] == "LSTM":
                model = LSTM(hyperparameters = hyperparameters)
                if inference == False:
                    optimiser = torch_optim_Adam(params = model.parameters(), lr = hyperparameters["learning_rate"])
                
            stats = checkpoint["stats"]
            model.load_state_dict(checkpoint["model"]["model_state_dict"])
            if inference == False:
                optimiser.load_state_dict(checkpoint["model"]["optimiser_state_dict"]) 
                # Only retrieve data if continuing training or testing on the test set
                self.DH_REF.retrieve_data(
                                        tickers = ["aapl", "tsla", "amzn", "goog", "msft", "googl"],
                                        start_date = "2015-01-01", # Date in MM/DD/YYYY or YYYY-MM-DD format, including the starting date
                                        end_date = "2020-01-01", # Not including the end_date
                                        interval = "1d",
                                        dated_sentiments = self.TDH_REF.dated_sentiments if hyperparameters["uses_dated_sentiments"] else None, # Dated sentiments for each company (None if not using)
                                        hyperparameters = hyperparameters
                                        )
            else:
                # Optimiser not declared yet so would run into error when returning
                optimiser = None
                # Set to evaluation mode at model inference
                model.eval()
        
        # Creates a new model
        else:
            if model_checkpoints_folder_exists == False:
                os_mkdir("model_checkpoints")
    
            # Model specifically for feature testing
            if feature_test_model_name:
                if os_path_exists("model_checkpoints/feature_test_models") == False:
                    os_mkdir("model_checkpoints/feature_test_models")
                self.clean_empty_directories()
                checkpoint_directory = f"model_checkpoints/feature_test_models/{feature_test_model_name}"
            # Model for deployment
            else:
                if os_path_exists("model_checkpoints/deployment_models") == False:
                    os_mkdir("model_checkpoints/deployment_models")
                self.clean_empty_directories()
                checkpoint_directory = f"model_checkpoints/deployment_models/{len(os_listdir('model_checkpoints/deployment_models'))}"

            os_mkdir(checkpoint_directory) 

            # Note: Use normalised data ("N") for RNN and standardised data ("S") for MLP 
            # Manual hyperparameters were not passed in when creating the new model (Use the recommended hyperparams)
            if manual_hyperparams == None:
                # Can change the values in this dictionary
                manual_hyperparams = {
                                    "architecture": "LSTM", 
                                    "batch_size": 32,
                                    "num_folds": 5,
                                    "multiplicative_trains": 2,
                                    "uses_dated_sentiments": True,
                                    "uses_single_sentiments": True, # Input = [Info1, Info2, Info3, Info4] + Single sentiment value on the date to predict
                                    "features_to_remove": [],
                                    "cols_to_alter": ["open", "close", "high", "low", "volume", "adjclose"],
                                    "rolling_periods": [2, 5, 10, 15, 20],
                                    "rolling_features": set(
                                                            [
                                                            "avg_open", 
                                                            "open_ratio", 
                                                            "avg_close", 
                                                            "close_ratio", 
                                                            "avg_volume", 
                                                            "volume_ratio", 
                                                            "trend_sum", 
                                                            "trend_mean", 
                                                            "close_diff", 
                                                            "close_diff_percentage", 
                                                            "rsi", 
                                                            "macd",
                                                            "bollinger_bands",
                                                            "average_true_range",
                                                            "stochastic_oscillator",
                                                            "on_balance_volume",
                                                            "ichimoku_cloud"
                                                            ]
                                                            ),
                                    "transform_after": True, # True to transform the comapnies data together or False for separately
                                    "train_split_decimal": 0.8, # Size of the train split as a decimal (0.8 = 80%)
                                    "train_data_params": None, # Training data parameters (mean, std, etc)
                                    }

                # Suggested values for the following hyperparameters, based on the model architecture
                if manual_hyperparams["architecture"] == "RNN":
                    manual_hyperparams["learning_rate"] = 1e-3
                    manual_hyperparams["N_OR_S"] = "N"
                    manual_hyperparams["num_context_days"] = 10

                elif manual_hyperparams["architecture"] == "MLP":
                    manual_hyperparams["learning_rate"] = 1e-4
                    manual_hyperparams["N_OR_S"] = "S"
                    manual_hyperparams["num_context_days"] = 1
                    
                elif manual_hyperparams["architecture"] == "LSTM":
                    manual_hyperparams["learning_rate"] = 1e-3
                    manual_hyperparams["N_OR_S"] = "N"
                    manual_hyperparams["num_context_days"] = 10
                
            # Removing any features to remove from the manual parameters
            """
            - Occurs if any feature is in both manual_hyperparams["features_to_remove"] and manual_hyperparameters["cols_to_alter"]
            - Only happens if the user feeds e.g. 
            "features_to_remove": ["adjclose"] 
            "cols_to_alter": ["open", "close", "high", "adjclose", "low", "volume"]
            """
            manual_hyperparams["cols_to_alter"] = [col for col in manual_hyperparams["cols_to_alter"] if col not in set(manual_hyperparams["features_to_remove"])]

            # Retrieve DH data
            self.DH_REF.retrieve_data(
                                    tickers = ["aapl", "tsla", "amzn", "goog", "msft", "googl"],
                                    start_date = "2015-01-01", # Date in MM/DD/YYYY or YYYY-MM-DD format, including the starting date
                                    end_date = "2020-01-01", # Not including the end_date
                                    interval = "1d",
                                    dated_sentiments = self.TDH_REF.dated_sentiments if manual_hyperparams["uses_dated_sentiments"] else None, # Dated sentiments for each company (None if not using)
                                    hyperparameters = manual_hyperparams
                                    )
            
            # Initialising the model and optimiser
            manual_hyperparams["n_features"] = self.DH_REF.n_features
            manual_hyperparams["fold_number"] = 0
            manual_hyperparams["train_data_params"] = self.DH_REF.train_data_params if manual_hyperparams["transform_after"] == True else None # The mean, std, maximums, minimums, depending on whether the training data for all the companies were transformed together, or separately
            
            if manual_hyperparams["architecture"] == "RNN":
                model = RNN(initial_in = self.DH_REF.n_features, final_out = 2, N_OR_S = manual_hyperparams["N_OR_S"], uses_single_sentiments = manual_hyperparams["uses_single_sentiments"])
                optimiser = torch_optim_Adam(params = model.parameters(), lr = manual_hyperparams["learning_rate"])

            elif manual_hyperparams["architecture"] == "MLP":
                model = MLP(initial_in = self.DH_REF.n_features, final_out = 2, N_OR_S = manual_hyperparams["N_OR_S"], uses_single_sentiments = manual_hyperparams["uses_single_sentiments"])
                optimiser = torch_optim_SGD(params = model.parameters(), lr = manual_hyperparams["learning_rate"])

            elif manual_hyperparams["architecture"] == "LSTM":
                model = LSTM(hyperparameters = manual_hyperparams)
                optimiser = torch_optim_Adam(params = model.parameters(), lr = manual_hyperparams["learning_rate"])

            # Modify hyperparams
            del manual_hyperparams["architecture"] # Not needed in hyperparameters dict
            hyperparameters = manual_hyperparams
            print(hyperparameters.keys())

            # Stats 
            metrics_used = ["loss", "accuracy", "precision", "recall", "f1"]
            stats_start = ["train_", "val_", "fold_t_", "fold_v_"] # train_loss_i, val_loss_i, fold_t_loss, fold_v_loss etc..
            stats = {f"{result_start}{metric}" + (("_i") if i == 0 or i == 1 else ""): [] for i, result_start in enumerate(stats_start) for metric in metrics_used}
            print(stats)

        # Move model and optimiser to GPU if possible
        if self.device != "cpu":
            model.to(device = self.device)

            if optimiser: # Optimiser was created
                for param in optimiser.state.values():
                    if isinstance(param, torch_Tensor):
                        param.data = param.data.to(self.device)
                        if param._grad is not None:
                            param._grad.data = param._grad.data.to(self.device)
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch_Tensor):
                                subparam.data = subparam.data.to(self.device)
                                if subparam._grad is not None:
                                    subparam._grad.data = subparam._grad.data.to(self.device)
                            
        # for param in optimiser.state.values():
        #     if isinstance(param, torch_Tensor):
        #         print(param.data.device)
        #     elif isinstance(param, dict):
        #         for subparam in param.values():
        #             if isinstance(subparam, torch_Tensor):
        #                 print(subparam.data.device)
        #                 if subparam._grad is not None:
        #                     print(subparam._grad.data.device)

        return model, optimiser, hyperparameters, stats, checkpoint_directory
    
    def clean_empty_directories(self):
        # Removes any empty model directories in the subdirectories of the model_checkpoints directory

        model_checkpoint_path = 'model_checkpoints/deployment_models'
        models_directory = os_listdir(model_checkpoint_path)
        for directory_name in models_directory:
            model_directory_path = f"{model_checkpoint_path}/{directory_name}"
            if len(os_listdir(model_directory_path)) == 0:
                os_rmdir(model_directory_path)

        model_checkpoint_path = 'model_checkpoints/feature_test_models'
        models_directory = os_listdir(model_checkpoint_path)
        for directory_name in models_directory:
            model_directory_path = f"{model_checkpoint_path}/{directory_name}"
            if len(os_listdir(model_directory_path)) == 0:
                os_rmdir(model_directory_path)