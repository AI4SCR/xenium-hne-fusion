from torch import nn


class Head(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 num_hidden_layers: int = 0, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = output_dim
        if num_hidden_layers == 0:
            self.head = nn.Sequential(nn.Linear(input_dim, output_dim))
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(num_hidden_layers):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)
