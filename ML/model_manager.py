
from os import listdir as os_listdir
from os import mkdir as os_mkdir
from os.path import exists as os_path_exists
from torch import load as torch_load
from torch.optim import Adam as torch_optim_Adam
from torch.optim import SGD as torch_optim_SGD
from models import *
from torch import Tensor as torch_Tensor

class ModelManager:

    def __init__(self, device, DH_reference, TDH_reference):

        self.device = device

        self.DH_REF = DH_reference
        self.TDH_REF = TDH_reference

    def initiate_model(self, model_number_load = None, manual_hyperparams = None):
        
        model_checkpoints_folder_exists = os_path_exists("model_checkpoints") 
        # Load existing model
        if model_number_load != None and model_checkpoints_folder_exists:
            checkpoint_directory = f"model_checkpoints/{model_number_load}"

        # Creates a new model
        else:
            if model_checkpoints_folder_exists == False:
                os_mkdir("model_checkpoints")

            checkpoint_directory = f"model_checkpoints/{len(os_listdir('model_checkpoints'))}"
            os_mkdir(checkpoint_directory) 

        if model_number_load != None and model_checkpoints_folder_exists:
            print("Loading existing model")
            existing_checkpoint_path = f"{checkpoint_directory}/fold_{len(os_listdir(f'{checkpoint_directory}')) - 1}.pth"
            checkpoint = torch_load(existing_checkpoint_path) # Load the last checkpoint (Which would be the complete model)
            print(existing_checkpoint_path)
            print(checkpoint.keys())

            """ 
            checkpoint is a dict consisting of 3 keys: 
            - "model" - Model + Optimiser state dicts and architecture used
            - "hyperparameters" - Hyperparameters used to train the model
            - "stats": Model statistics (e.g. loss, accuracy, f1 score, etc..)
            """

            hyperparameters = checkpoint["hyperparameters"] # Load hyperparameters of the saved model

            if checkpoint["model"]["architecture"] == "RNN":
                model = RNN(initial_in = hyperparameters["n_features"], final_out = 2, N_OR_S = hyperparameters["N_OR_S"])
                optimiser = torch_optim_Adam(params = model.parameters(), lr = hyperparameters["learning_rate"])
            
            elif checkpoint["model"]["architecture"] == "MLP":
                model = MLP(initial_in = hyperparameters["n_features"], final_out = 2, N_OR_S = hyperparameters["N_OR_S"])
                optimiser = torch_optim_SGD(params = model.parameters(), lr = hyperparameters["learning_rate"])
            
            model.load_state_dict(checkpoint["model"]["model_state_dict"])
            optimiser.load_state_dict(checkpoint["model"]["optimiser_state_dict"]) 
            stats = checkpoint["stats"]

            # Retrieve DH data
            self.DH_REF.retrieve_data(
                                    tickers = ["aapl", "tsla", "amzn", "goog", "msft", "googl"],
                                    start_date = "1/01/2015",
                                    end_date = "31/12/2019", 
                                    interval = "1d",
                                    transform_after = True,
                                    dated_sentiments = self.TDH_REF.dated_sentiments if hyperparameters["uses_dated_sentiments"] else None # Dated sentiments for each company (None if not using)
                                    )
        else:
            # Note: Use normalised data ("N") for RNN and standardised data ("S") for MLP 

            # Manual hyperparameters were not passed in when creating the new model (Use the recommended hyperparams)
            if manual_hyperparams == None:
                # Can change the values in this dictionary
                manual_hyperparams = {
                                    "architecture": "RNN", 
                                    "batch_size": 32,
                                    "num_folds": 5,
                                    "multiplicative_trains": 2,
                                    "uses_dated_sentiments": True,
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

            # Retrieve DH data
            self.DH_REF.retrieve_data(
                                    tickers = ["aapl", "tsla", "amzn", "goog", "msft", "googl"],
                                    start_date = "1/01/2015",
                                    end_date = "31/12/2019", 
                                    interval = "1d",
                                    transform_after = True,
                                    dated_sentiments = self.TDH_REF.dated_sentiments if manual_hyperparams["uses_dated_sentiments"] else None # Dated sentiments for each company (None if not using)
                                    )
            
            # Initialising the model and optimiser
            if manual_hyperparams["architecture"] == "RNN":
                model = RNN(initial_in = self.DH_REF.n_features, final_out = 2, N_OR_S = manual_hyperparams["N_OR_S"])
                optimiser = torch_optim_Adam(params = model.parameters(), lr = manual_hyperparams["learning_rate"])

            elif manual_hyperparams["architecture"] == "MLP":
                model = MLP(initial_in = self.DH_REF.n_features, final_out = 2, N_OR_S = manual_hyperparams["N_OR_S"])
                optimiser = torch_optim_SGD(params = model.parameters(), lr = manual_hyperparams["learning_rate"])

            # Modify hyperparams
            del manual_hyperparams["architecture"] # Not needed in hyperparameters dict
            hyperparameters = manual_hyperparams
            hyperparameters["fold_number"] = 0
            hyperparameters["n_features"] = self.DH_REF.n_features
            print(hyperparameters.keys())

            # Stats 
            metrics_used = ["loss", "accuracy", "precision", "recall", "f1"]
            stats_start = ["train_", "val_", "fold_t_", "fold_v_"] # train_loss_i, val_loss_i, fold_t_loss, fold_v_loss etc..
            stats = {f"{result_start}{metric}" + (("_i") if i == 0 or i == 1 else ""): [] for i, result_start in enumerate(stats_start) for metric in metrics_used}
            print(stats)
    
        # Move the model and optimiser to CPU / GPU
        model.to(device = self.device)
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
