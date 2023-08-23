
from typing import Any
import torch.nn as nn
from torch import zeros as torch_zeros

class MLP(nn.Module):

    def __init__(self, initial_in, final_out, N_OR_S):

        super(MLP, self).__init__()
        
        # Cross validation results (Epochs = 25000, num_folds = 10)
        
        # Normalised data
        # (2) TAccuracy: 51.40895833333333) | VAccuracy: 50.34202777777778 | TLoss: 0.7012315288949014 | VLoss: 0.7085194245812628 | TPrecision: 0.5025283826230803 | VPrecision: 0.5102507867960342 | TRecall: 0.5021342522499035 | VRecall 0.7309120488661836 | TF1: 0.4880115095955318 | VF1: 0.5900783530057129 
        # (3) TAccuracy: 51.55272222222222) | VAccuracy: 49.284 | TLoss: 0.6994038878783914 | VLoss: 0.7200308687763745 | TPrecision: 0.5038663937409447 | VPrecision: 0.5069324597900078 | TRecall: 0.5088018969980461 | VRecall 0.5958059184476738 | TF1: 0.49192464056477697 | VF1: 0.5228271428565807

        # Standardised data
        # (2) TAccuracy: 51.34845833333334) | VAccuracy: 50.619111111111096 | TLoss: 0.7015705746686458 | VLoss: 0.7103137350401613 | TPrecision: 0.5021654293802416 | VPrecision: 0.5157297765442445 | TRecall: 0.5112518890723601 | VRecall 0.6271348355543205 | TF1: 0.49455508367829737 | VF1: 0.5548726289603318 
        # (3) TAccuracy: 52.05580555555556) | VAccuracy: 49.68394444444445 | TLoss: 0.6988001317003039 | VLoss: 0.7179158999962277 | TPrecision: 0.5101338190865925 | VPrecision: 0.5139042559802637 | TRecall: 0.5020571426920003 | VRecall 0.6067671287374661 | TF1: 0.49349724992391664 | VF1: 0.5376996089209956    

        self.model = nn.Sequential( 
                                    # -----------------------------------------------------------------
                                    # Config 2:

                                    nn.Linear(in_features = initial_in, out_features = initial_in // 2),
                                    nn.BatchNorm1d(num_features = initial_in // 2),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in // 2, out_features = initial_in // 4),
                                    nn.BatchNorm1d(num_features = initial_in // 4),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in // 4, out_features = final_out)
                                    
                                    # -----------------------------------------------------------------
                                    # Config 3:

                                    # nn.Linear(in_features = initial_in, out_features = initial_in * 2),
                                    # nn.BatchNorm1d(num_features = initial_in * 2),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in * 2, out_features = initial_in * 2),
                                    # nn.BatchNorm1d(num_features = initial_in * 2),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in * 2, out_features = initial_in),
                                    # nn.BatchNorm1d(num_features = initial_in),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in, out_features = initial_in // 2),
                                    # nn.BatchNorm1d(num_features = initial_in // 2),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in // 2, out_features = final_out)
                                    
                                    )
        
        self.initialise_weights(non_linearity = "relu")
        
        # Determines whether the model will use normalised or standardised data for training and inference
        self.N_OR_S = N_OR_S
    
    def __call__(self, inputs):
        return self.model(inputs)

    def initialise_weights(self, non_linearity):
        
        # Uses Kai Ming uniform for ReLU activation functions, but Kai Ming normal for other activation functions
        init_function = nn.init.kaiming_uniform_ if non_linearity == "relu" else nn.init.kaiming_normal_
        
        # Apply Kai-Ming initialisation to all linear layer weights
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init_function(layer.weight, mode = "fan_in", nonlinearity = non_linearity)

class RNN(nn.Module):

    def __init__(self, initial_in, final_out, N_OR_S):
        super(RNN, self).__init__()

        # Cross validation results (Epochs = 5000, num_folds = 10)

        # Normalised data
        # (1) TAccuracy: 57.44354166666666) | VAccuracy: 54.31861111111112 | TLoss: 0.6760456508689456 | VLoss: 0.7043166189339425 | TPrecision: 0.5758899828110903 | VPrecision: 0.5400229215866872 | TRecall: 0.5345691893685933 | VRecall 0.5903054536139032 | TF1: 0.5416557570162557 | VF1: 0.5394078170391472    
        # (2) TAccuracy: 57.39618055555556) | VAccuracy: 53.26263888888889 | TLoss: 0.6748167537026935 | VLoss: 0.6937137216462029 | TPrecision: 0.5792144796145935 | VPrecision: 0.541103975897471 | TRecall: 0.5192639991477738 | VRecall 0.5236433334519626 | TF1: 0.5311361625404567 | VF1: 0.5082877600090694

        # Standardised data
        # (1) TAccuracy: 57.8063888888889) | VAccuracy: 54.16430555555556 | TLoss: 0.6744753837863603 | VLoss: 0.7034466716210048 | TPrecision: 0.5799920688806193 | VPrecision: 0.537433772745333 | TRecall: 0.5397820134660448 | VRecall 0.5859260845369195 | TF1: 0.5468734678021602 | VF1: 0.5341009027190929
        # (2) TAccuracy: 57.66645833333333) | VAccuracy: 53.292986111111105 | TLoss: 0.6741172053244379 | VLoss: 0.6992408750971157 | TPrecision: 0.5784034473296962 | VPrecision: 0.5323862838852966 | TRecall: 0.5455958952027022 | VRecall 0.5749857534019637 | TF1: 0.54754735092235 | VF1: 0.5214294278356074

        self.layers = nn.Sequential(
                                    # 1
                                    # nn.Linear(initial_in, initial_in),
                                    # nn.BatchNorm1d(num_features = initial_in),
                                    # nn.ReLU(),

                                    # nn.Linear(initial_in , initial_in // 2),
                                    # nn.BatchNorm1d(num_features = initial_in // 2),
                                    # nn.ReLU(),

                                    # nn.Linear(initial_in  // 2, initial_in // 4),
                                    # nn.BatchNorm1d(num_features = initial_in // 4),
                                    # nn.ReLU(),

                                    # 2
                                    nn.Linear(initial_in, initial_in),
                                    nn.BatchNorm1d(num_features = initial_in),
                                    nn.ReLU(),

                                    nn.Linear(initial_in, initial_in),
                                    nn.BatchNorm1d(num_features = initial_in),
                                    nn.ReLU(),

                                    nn.Linear(initial_in , initial_in // 2),
                                    nn.BatchNorm1d(num_features = initial_in // 2),
                                    nn.ReLU(),

                                    nn.Linear(initial_in // 2, initial_in // 2),
                                    nn.BatchNorm1d(num_features = initial_in // 2),
                                    nn.ReLU(),

                                    nn.Linear(initial_in  // 2, initial_in // 4),
                                    nn.BatchNorm1d(num_features = initial_in // 4),
                                    nn.ReLU(),
                                    )
        # Hidden state layer
        self.hidden_layer = nn.Linear(initial_in // 4, out_features = initial_in // 4, bias = True)
        self.hidden_state_af = nn.ReLU()

        self.O = nn.Linear(initial_in // 4, final_out)

        self.initialise_weights(non_linearity = "relu")

        # Determines whether the model will use normalised or standardised data for training and inference
        self.N_OR_S = N_OR_S

    def __call__(self, inputs):
        
        # inputs.shape = [Number of x consecutive day sequences, x consecutive day sequences, num features in a single day]

        # Let num_context_days = 10, batch_size = 32
        # Single batch should be [10 x [32 * num_features] ]
        # 32 x [ClosingP, OpeningP, Volume, etc..] 10 days ago
        # The next batch for the recurrence will be the day after that day
        # 32 x [ClosingP, OpeningP, Volume, etc..] 9 days ago
        # Repeats until all 10 days have been passed in (for a single batch)
        
        # Forward pass should proceed as:
        # [batch_size, 0th day, sequence features]
        # [batch_size, 1st day, sequence features]
        # [batch_size, 2nd day, sequence features]

        num_context_days = inputs.shape[0]
        batch_size = inputs.shape[1]
        self.hidden_state = torch_zeros(batch_size, self.hidden_layer.weight.shape[1], device = self.O.weight.device) # Initialise hidden state at the start of each forward pass as zeroes

        # Recurrence:
        for i in range(num_context_days):
            # Current context day with batch_size examples
            current_day_batch = inputs[i][:][:]
            
            # Pass through all layers except the output layer
            output = self.layers(current_day_batch)

            # Pass the previous hidden state and current output (after the linear layers) into the hidden layer and then the activation function for the hidden layer
            """Note:
            - If this is the first training step, the hidden state would be all zeroes. This means that it would have no effect on the output for this step.
            - If this isn't the first training step, the hidden state will be added with the current output after the linear layers
            """
            self.hidden_state = self.hidden_state_af(self.hidden_layer(self.hidden_state + output))
    
        # Predict output for the "num_context_days + 1"th day 
        # Note: Shape should be [batch_size, 2], describing the probability assigned (after softmax) that the stock price will go up or down for each sequence in the batch
        output = self.O(self.hidden_state)

        return output
    
    def initialise_weights(self, non_linearity):
        
        # Uses Kai Ming uniform for ReLU activation functions, but Kai Ming normal for other activation functions
        init_function = nn.init.kaiming_uniform_ if non_linearity == "relu" else nn.init.kaiming_normal_
        
        # Apply Kai-Ming initialisation to all linear layer weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init_function(layer.weight, mode = "fan_in", nonlinearity = non_linearity)
        init_function(self.hidden_layer.weight, mode = "fan_in", nonlinearity = non_linearity)
        init_function(self.O.weight, mode = "fan_in", nonlinearity = non_linearity)