
from typing import Any
import torch.nn as nn
from torch import zeros as torch_zeros
from torch import concat as torch_concat

class MLP(nn.Module):

    def __init__(self, initial_in, final_out, N_OR_S, uses_single_sentiments):

        super(MLP, self).__init__()
        
        self.model = nn.Sequential( 
                                    # -----------------------------------------------------------------
                                    # Config 2:

                                    nn.Linear(in_features = initial_in, out_features = initial_in // 2),
                                    nn.BatchNorm1d(num_features = initial_in // 2),
                                    nn.ReLU(),

                                    nn.Linear(in_features = initial_in // 2, out_features = initial_in // 4),
                                    nn.BatchNorm1d(num_features = initial_in // 4),
                                    nn.ReLU()
                                    
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
        
        # Output layer
        self.O = nn.Linear(in_features = (initial_in // 4) + uses_single_sentiments, out_features = final_out)

        self.initialise_weights(non_linearity = "relu")
        
        # Determines whether the model will use normalised or standardised data for training and inference
        self.N_OR_S = N_OR_S
    
    def forward(self, inputs, single_sentiment_values):

        if single_sentiment_values != None:
            # - Concatenate outputs after hidden layers with the single sentiment values
            # - Pass through output layer
            return self.O(torch_concat([self.model(inputs), single_sentiment_values], dim = 1))
        else:
            # Pass through output layer directly after hidden layers
            return self.O(self.model(inputs))

    def initialise_weights(self, non_linearity):
        
        # Uses Kai Ming uniform for ReLU activation functions, but Kai Ming normal for other activation functions
        init_function = nn.init.kaiming_uniform_ if non_linearity == "relu" else nn.init.kaiming_normal_
        
        # Apply Kai-Ming initialisation to all linear layer weights
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init_function(layer.weight, mode = "fan_in", nonlinearity = non_linearity)
        init_function(self.O.weight, mode = "fan_in", nonlinearity = non_linearity)

class RNN(nn.Module):

    def __init__(self, initial_in, final_out, N_OR_S, uses_single_sentiments):
        super(RNN, self).__init__()

        self.layers = nn.Sequential(
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
        
        # Output layer
        self.O = nn.Linear((initial_in // 4) + uses_single_sentiments, final_out) # + 1 if using the sentiment from the date to predict

        self.initialise_weights(non_linearity = "relu")

        # Determines whether the model will use normalised or standardised data for training and inference
        self.N_OR_S = N_OR_S

    def forward(self, inputs, single_sentiment_values):
        
        """
        Forward pass notes:
        - inputs.shape = [Number of x context days in a data sequence, batch size, num features in each day]

        # Recurrent layer explained:
        Let num_context_days = 10, batch_size = 32, num_features = 15
        A Single batch should be [10, 32, 15]
        At each timestep in the recurrence, it extracts a single batch (i.e. [1, 32, 15] ---> [32, 15])
        
        This batch (i.e. [32, 15]) will then be passed into the model
        timestep = 0 (10 days ago) = 32 x [ClosingP, OpeningP, Volume, etc..]
        The hidden state is then updated by passing the addition of the output after the RNN layer (self.layers) and the hidden state from the previous time step into the hidden state layer + activation.
        timestep = 1 (9 days ago) = 32 x [ClosingP, OpeningP, Volume, etc..]
        The hidden state is updated again (Done in the same with the previous hidden state)
        This is repeated until all batches for all 10 days have been passed into the model

        # Final output
        
        If the model uses single sentiments (i.e. a batch of single sentiments from the days to predict the stock trends):
        - The single sentiments are concatenated with the output of the model after the recurrent layer: i.e. [32, 1] concatenated with [32, num_features_after_recurrence] --> [32, num_features_after_recurrence + 1] 
        - The output after the recurrence is passed through the output layer: i.e. [32, num_features_after_recurrence + 1] ---> [32, 2]

        If the model does not use single sentiments:
        - The output after the recurrence is passed through the output layer: i.e. [32, num_features_after_recurrence] ---> [32, 2]

        The final output is a probability distribution for each of the 32 data sequences in the batch: [Probability the stock goes down, Probability the stock goes up]
        """
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

        # Using input as [data_sequence] + single sentiment value on the day to predict
        if single_sentiment_values != None:
            # single_sentiment_values.shape = [batch_size, 1]
            # self.hidden_state.shape = [batch_size, self.hidden_state.shape[1]]
            self.hidden_state = torch_concat([self.hidden_state, single_sentiment_values], dim = 1)
        
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

class LSTM(nn.Module):
    
    def __init__(self, hyperparameters):
        super(LSTM, self).__init__()

        self.N_OR_S = hyperparameters["N_OR_S"]
        self.batch_size = hyperparameters["batch_size"]
        self.n_features = hyperparameters["n_features"]
        initial_in = hyperparameters["n_features"] # Should be the same as the number of hidden units in the hidden state
        
        # LSTM layers
        # Note: Don't use : "LSTMLayer()] * n_lstm_layers" as it will create a list of layers sharing the same instance of LSTMLayer, sharing the same weights + biases for each of its components as well
        self.lstm_layers = nn.ModuleList([LSTMLayer(hyperparameters = hyperparameters)])
                                    
        # Linear layers
        self.layers = nn.Sequential(
                                    nn.Linear(initial_in, initial_in),
                                    nn.BatchNorm1d(num_features = initial_in),
                                    nn.ReLU(),

                                    nn.Linear(initial_in , initial_in // 2),
                                    nn.BatchNorm1d(num_features = initial_in // 2),
                                    nn.ReLU(),

                                    nn.Linear(initial_in  // 2, initial_in // 4),
                                    nn.BatchNorm1d(num_features = initial_in // 4),
                                    nn.ReLU(),
                                    )
        self.output_layer = nn.Linear((initial_in  // 4) + hyperparameters["uses_single_sentiments"], 2)

        # Initialise weights for all model weights
        self.initialise_weights()
    
    def forward(self, inputs, single_sentiment_values):

        # Initialise hidden state / short term memory at the start of each forward pass as zeroes
        """
        - Specify the device as this is initialised at the start of every forward pass (would cause an error between tensors being on different devices if using CUDA)
        - Set requires_grad to False after (as passing requires_grad = False does not work)
        """
        self.short_term_memory = nn.Parameter(torch_zeros(self.batch_size, self.n_features, requires_grad = False, device = self.output_layer.weight.device)) 
        self.short_term_memory.requires_grad = False

        num_context_days = inputs.shape[0]

        # For each time step / context day
        for i in range(num_context_days):
            # Current context day with batch_size examples
            current_day_batch = inputs[i][:][:]

            # Pass through the LSTM layers, which will pass the inputs through the LSTM cells, updating the short term memory and long-term memory
            """
            Pass inputs and previous hidden state (0) into LSTM layer 1
            Update hidden state (1)

            Pass inputs and previous hidden state (1) into LSTM layer 2
            Update hidden state (2)

            etc...
            """
            for lstm_layer in self.lstm_layers:
                """
                - Update the hidden state (short term memory) after passing through each LSTM layer
                - Only update the data of the short term memory, without modifying the requires_grad attribute of the parameter
                """ 
                self.short_term_memory.data = lstm_layer(inputs = current_day_batch, hidden_state = self.short_term_memory).data
            
        # Pass through linear layers (excluding output layer) to reduce dimensionality
        output = self.layers(self.short_term_memory)
        
        # Single sentiment values to concatenate
        if single_sentiment_values != None:
            output = torch_concat([output, single_sentiment_values], dim = 1)
            
        # Pass through output layer
        return self.output_layer(output)

    def initialise_weights(self):
        # Initialises all weights for the LSTM cells and gates of each LSTM layer
        for lstm_layer in self.lstm_layers:
            lstm_layer.initialise_weights()

        # Apply Kai-Ming initialisation to all linear layer weights (using Kai Ming uniform because of ReLU activation functions)
        for layer in self.layers:
            init_function = nn.init.kaiming_uniform_
            if isinstance(layer, nn.Linear):
                init_function(layer.weight, mode = "fan_in", nonlinearity = "relu")
            init_function(self.output_layer.weight, mode = "fan_in", nonlinearity =  "relu")

class LSTMLayer(nn.Module):
    def __init__(self, hyperparameters):
        super(LSTMLayer, self).__init__()
        # LSTM Cells
        self.cells = nn.ModuleList([LSTMCell(hyperparameters = hyperparameters) for _ in range(hyperparameters["n_lstm_cells"])])
    
    def forward(self, inputs, hidden_state):
        # Pass through LSTM cells, updating the short term memory and long-term memory
        for cell in self.cells:
            hidden_state = cell(inputs = inputs, hidden_state = hidden_state)
        return hidden_state

    def initialise_weights(self):
        # Initialises all weights in the LSTM cells and gates using Xavier uniform (Because of tanh + sigmoid activations)
        for cell in self.cells:
            cell.initialise_weights(init_function = nn.init.xavier_uniform_)

class LSTMCell(nn.Module):
    def __init__(self, hyperparameters):
        super(LSTMCell, self).__init__()
        self.parameters_dict = nn.ParameterDict({
                                                "FG": ForgetGate(n_features = hyperparameters["n_features"]),
                                                "IG": InputGate(n_features = hyperparameters["n_features"]),
                                                "OG": OutputGate(n_features = hyperparameters["n_features"]),
                                                })
        self.cell_state = nn.Parameter(torch_zeros(hyperparameters["batch_size"], hyperparameters["n_features"]))
        # - Set requires_grad to False after (as passing requires_grad = False does not work)
        self.cell_state.requires_grad = False
    
    def forward(self, inputs, hidden_state):
        """
        Notes:
        (forget_gate_output * hidden_state) = How much information from the previous cell state should be retained
        (input_gate_output * output_gate_output) = How much new information should be added to the cell state
        - These operations are completed inside of the gates
        """
        forget_gate_output = self.parameters_dict["FG"](inputs = inputs, short_term_memory = hidden_state)
        input_gate_output = self.parameters_dict["IG"](inputs = inputs, short_term_memory = hidden_state)
        output_gate_output = self.parameters_dict["OG"](inputs = inputs, long_term_memory = self.cell_state, short_term_memory = hidden_state)

        # Update hidden state and cell state
        hidden_state = output_gate_output # (i.e. the new hidden state, which would be the output gate output)
        """
        Cell state:
        'cell_state = (self.cell_state * forget_gate_output) + input_gate_output' has been used to avoid in-place operations, but is the same as: 
        1. self.cell_state *= forget_gate_output
        2. self.cell_state += input_gate_output
        cell_state = (Previous cell state * Long term to remember percentage) (Potential memory to remember * Potential long term memory)

        Hidden state:
        'self.cell_state.data = cell_state.data' has been used to update the cell state data without modifying the requires_grad attribute of the cell state (Would cause an error)
        hidden_state = Potential short term memory * Potential memory to remember
        """
        cell_state = (self.cell_state * forget_gate_output) + input_gate_output 
        self.cell_state.data = cell_state.data

        # Return the new hidden state
        return hidden_state

    def initialise_weights(self, init_function):
        # Calls the initialise weights method for all gates
        self.parameters_dict["FG"].initialise_weights(init_function)
        self.parameters_dict["IG"].initialise_weights(init_function)
        self.parameters_dict["OG"].initialise_weights(init_function)

class ForgetGate(nn.Module):
    def __init__(self, n_features):
        super(ForgetGate, self).__init__()
        self.parameters_dict = nn.ParameterDict({
                                                "sigmoid": nn.Sigmoid(),
                                                "sigmoid_layer": nn.Linear(in_features = n_features, out_features = n_features, bias = True),
                                                })
    def forward(self, inputs, short_term_memory):
        """
        Long term to remember percentage = Sigmoid activation((Layer output + short_term_memory) + Nodebias0)
        Long term memory *= Long term to remember percentage
        """
        return self.parameters_dict["sigmoid"](self.parameters_dict["sigmoid_layer"](inputs + short_term_memory))
    
    def initialise_weights(self, init_function):
        init_function(self.parameters_dict["sigmoid_layer"].weight)

class InputGate(nn.Module):
    def __init__(self, n_features):
        super(InputGate, self).__init__()
        self.parameters_dict = nn.ParameterDict({
                                                "sigmoid": nn.Sigmoid(),
                                                "sigmoid_layer": nn.Linear(in_features = n_features, out_features = n_features, bias = True),
                                                "tanh": nn.Tanh(),
                                                "tanh_layer": nn.Linear(in_features = n_features, out_features = n_features, bias = True)
                                                })
    def forward(self, inputs, short_term_memory):
        """
        Potential memory to remember = Sigmoid activation((Layer output + short_term_memory) + Nodebias1)
        = (self.parameters_dict["sigmoid"](self.parameters_dict["sigmoid_layer"](inputs + short_term_memory))

        Potential long term memory = Tanh activation((Layer output + short_term_memory) + Nodebias2)
        = self.parameters_dict["tanh"](self.parameters_dict["tanh_layer"](inputs + short_term_memory))

        Long term memory += (Potential memory to remember * Potential long term memory)(Potential memory to remember * Potential long term memory)
        """
        return (self.parameters_dict["sigmoid"](self.parameters_dict["sigmoid_layer"](inputs + short_term_memory)) * self.parameters_dict["tanh"](self.parameters_dict["tanh_layer"](inputs + short_term_memory)))
    
    def initialise_weights(self, init_function):
        init_function(self.parameters_dict["sigmoid_layer"].weight)
        init_function(self.parameters_dict["tanh_layer"].weight)

class OutputGate(nn.Module):
    def __init__(self, n_features):
        super(OutputGate, self).__init__()
        self.parameters_dict = nn.ParameterDict({
                                                "sigmoid": nn.Sigmoid(),
                                                "sigmoid_layer": nn.Linear(in_features = n_features, out_features = n_features, bias = True),
                                                "tanh": nn.Tanh(),
                                                "tanh_layer": nn.Linear(in_features = n_features, out_features = n_features, bias = True)
                                                })

    def forward(self, inputs, long_term_memory, short_term_memory):
        """
        Potential short term memory = Tanh activation(Long term memory):
        = self.parameters_dict["tanh"](long_term_memory) 

        Potential memory to remember = Sigmoid activation((Layer output + short_term_memory) + Nodebias3):
        = self.parameters_dict["sigmoid"](self.parameters_dict["sigmoid_layer"](inputs + short_term_memory))

        New short term memory(The final output) = Potential short term memory * Potential memory to remember
        """
        return self.parameters_dict["tanh"](long_term_memory) * self.parameters_dict["sigmoid"](self.parameters_dict["sigmoid_layer"](inputs + short_term_memory))

    def initialise_weights(self, init_function):
        init_function(self.parameters_dict["sigmoid_layer"].weight)
        init_function(self.parameters_dict["tanh_layer"].weight)