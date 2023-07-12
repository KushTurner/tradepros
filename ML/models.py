
from typing import Any
import torch.nn as nn
from torch import stack as torch_stack

class MLP(nn.Module):

    def __init__(self, initial_in, final_out):

        super(MLP, self).__init__()
        
        # Accuracies tested (32 batch size, 200_000 steps)
        
        # Unnormalised + Unstandardised data
        # (1) TrainAccuracy(%): 52.349500000000006 | ValAccuracy(%): 53.997499999999995
        # (2) TrainAccuracy(%): 52.05499999999999 | ValAccuracy(%): 54.777
        # (3) TrainAccuracy(%): 52.3425 | ValAccuracy(%): 54.234

        # Normalised data:
        # (1) TrainAccuracy(%): 56.308499999999995 | ValAccuracy(%): 53.5775
        # (2) TrainAccuracy(%): 51.9335 | ValAccuracy(%): 54.456
        # (3) TrainAccuracy(%): 55.704 | ValAccuracy(%): 55.306999999999995
        
        # Standardised data:
        # (1) TrainAccuracy(%): 56.381499999999996 | ValAccuracy(%): 51.82600000000001
        # (2) TrainAccuracy(%): 51.4425 | ValAccuracy(%): 51.687000000000005
        # (3) TrainAccuracy(%): 55.01200000000001 | ValAccuracy(%): 54.623999999999995
        
        self.model = nn.Sequential( 
                                    # -----------------------------------------------------------------
                                    # Config 1:

                                    # nn.Linear(in_features = initial_in, out_features = initial_in * 2),
                                    # nn.BatchNorm1d(num_features = initial_in * 2),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in * 2, out_features = initial_in * 2),
                                    # nn.BatchNorm1d(num_features = initial_in * 2),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in * 2, out_features = initial_in * 2),
                                    # nn.BatchNorm1d(num_features = initial_in * 2),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in * 2, out_features = initial_in),
                                    # nn.BatchNorm1d(num_features = initial_in),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in, out_features = final_out)

                                    # -----------------------------------------------------------------
                                    # Config 2:

                                    # nn.Linear(in_features = initial_in, out_features = initial_in // 2),
                                    # nn.BatchNorm1d(num_features = initial_in // 2),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in // 2, out_features = initial_in // 4),
                                    # nn.BatchNorm1d(num_features = initial_in // 4),
                                    # nn.ReLU(),

                                    # nn.Linear(in_features = initial_in // 4, out_features = final_out)
                                    
                                    # -----------------------------------------------------------------
                                    # Config 3:

                                    nn.Linear(in_features = initial_in, out_features = initial_in * 2),
                                    nn.BatchNorm1d(num_features = initial_in * 2),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in * 2, out_features = initial_in * 2),
                                    nn.BatchNorm1d(num_features = initial_in * 2),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in * 2, out_features = initial_in),
                                    nn.BatchNorm1d(num_features = initial_in),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in, out_features = initial_in // 2),
                                    nn.BatchNorm1d(num_features = initial_in // 2),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in // 2, out_features = final_out)
                                    
                                    )
        
        self.initialise_weights(non_linearity = "relu")
    
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

    def __init__(self, initial_in, final_out):
        super(RNN, self).__init__()

        self.L1 = nn.Linear(initial_in, initial_in)
        self.L2 = nn.Linear(initial_in , initial_in // 2)
        self.L3 = nn.Linear(initial_in  // 2, initial_in // 4)

        self.layers = [layer for layer in self._modules.values()] # All layers except output layer

        self.O = nn.Linear(initial_in // 4, final_out)

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

        # Recurrence:
        for i in range(num_context_days):
            # Current context day with batch_size examples
            current_day_batch = inputs[i][:][:]
            
            # Pass through all layers except the output layer
            output = self.forward(inputs = current_day_batch)

        # After all days, find and return the output
        return self.O(output)

    def forward(self, inputs):
        
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
