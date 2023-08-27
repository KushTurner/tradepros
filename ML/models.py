
from typing import Any
import torch.nn as nn
from torch import zeros as torch_zeros

class MLP(nn.Module):

    def __init__(self, initial_in, final_out, N_OR_S):

        super(MLP, self).__init__()
        
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

                                    # # 3 (Test config)

                                    # nn.Linear(initial_in, initial_in),
                                    # nn.BatchNorm1d(num_features = initial_in),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),
                                    
                                    # nn.Linear(initial_in, initial_in),
                                    # nn.BatchNorm1d(num_features = initial_in),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in, initial_in * 2),
                                    # nn.BatchNorm1d(num_features = initial_in * 2),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in * 2, initial_in * 2),
                                    # nn.BatchNorm1d(num_features = initial_in * 2),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in * 2, initial_in * 2),
                                    # nn.BatchNorm1d(num_features = initial_in * 2),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in * 2, initial_in),
                                    # nn.BatchNorm1d(num_features = initial_in),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in, initial_in),
                                    # nn.BatchNorm1d(num_features = initial_in),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in , initial_in // 2),
                                    # nn.BatchNorm1d(num_features = initial_in // 2),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in // 2, initial_in // 2),
                                    # nn.BatchNorm1d(num_features = initial_in // 2),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in // 2, initial_in // 2),
                                    # nn.BatchNorm1d(num_features = initial_in // 2),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in // 2, initial_in // 2),
                                    # nn.BatchNorm1d(num_features = initial_in // 2),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in // 2, initial_in // 2),
                                    # nn.BatchNorm1d(num_features = initial_in // 2),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),
                                    
                                    # nn.Linear(initial_in // 2, initial_in // 2),
                                    # nn.BatchNorm1d(num_features = initial_in // 2),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in  // 2, initial_in // 4),
                                    # nn.BatchNorm1d(num_features = initial_in // 4),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),

                                    # nn.Linear(initial_in  // 4, initial_in // 4),
                                    # nn.BatchNorm1d(num_features = initial_in // 4),
                                    # nn.ReLU(),
                                    # nn.Dropout(p = 0.2),
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