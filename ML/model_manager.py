
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

    def initiate_model(self, model_number_load = None):
        
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
            - "results": Results from training the model
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
            _uses_dated_sentiments = True

            # Retrieve DH data
            self.DH_REF.retrieve_data(
                                    tickers = ["aapl", "tsla", "amzn", "goog", "msft", "googl"],
                                    start_date = "1/01/2015",
                                    end_date = "31/12/2019", 
                                    interval = "1d",
                                    transform_after = True,
                                    dated_sentiments = self.TDH_REF.dated_sentiments if _uses_dated_sentiments else None # Dated sentiments for each company (None if not using)
                                    )
            
            _learning_rate = 1e-3
            model = RNN(initial_in = self.DH_REF.n_features, final_out = 2, N_OR_S = "N")
            optimiser = torch_optim_Adam(params = model.parameters(), lr = _learning_rate)

            # _learning_rate = 0.0001
            # model = MLP(initial_in = self.DH_REF.n_features, final_out = 2, N_OR_S = "S")
            # optimiser = torch_optim_SGD(params = model.parameters(), lr = _learning_rate)
            
            # Hyperparameters
            _n_features = self.DH_REF.n_features
            _N_OR_S = model.N_OR_S
            _batch_size = 32
            _num_context_days = 10 if isinstance(model, RNN) else 1 # Number of days used as context (Used for RNN)
            _num_folds = 5 # Number of folds used in cross-validation
            _fold_number = 0
            _multiplicative_trains = 1
            hyperparameters = {key[1:]:val for key, val in locals().items() if key.startswith("_")}
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
