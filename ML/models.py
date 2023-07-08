
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, initial_in, final_out):

        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
                                    nn.Linear(in_features = initial_in, out_features = initial_in * 2),
                                    nn.BatchNorm1d(num_features = initial_in * 2),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in * 2, out_features = initial_in * 2),
                                    nn.BatchNorm1d(num_features = initial_in * 2),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in * 2, out_features = initial_in * 2),
                                    nn.BatchNorm1d(num_features = initial_in * 2),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in * 2, out_features = initial_in),
                                    nn.BatchNorm1d(num_features = initial_in),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in, out_features = final_out)
                                    )

    def __call__(self, inputs):
        return self.model(inputs)