
import torch.nn as nn

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