from torch import nn

class MLPRegressor(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, n_hidden:int,  output_size:int):
        super(MLPRegressor, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_hidden)])
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))

        x = self.output_layer(x)

        return x