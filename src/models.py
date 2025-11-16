import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, output_dim: int, 
                 num_layers: int = 3, activation: str = 'leaky_relu',
                 dropout: float = 0.1, logger=None, use_batchnorm: bool = False, use_layernorm: bool = True,
                 init_type: str = "xavier_uniform"):
        """Initializes a multi-layer perceptron with residual connections and optional normalization."""
        super(MLPModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm
        self.init_type = init_type

        # Input layer
        self.input_layer = nn.Linear(input_dim, hid_dim)
        self.input_bn = nn.BatchNorm1d(hid_dim) if use_batchnorm else nn.Identity()
        self.input_ln = nn.LayerNorm(hid_dim) if use_layernorm else nn.Identity()

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        self.hidden_lns = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hid_dim, hid_dim))
            self.hidden_bns.append(nn.BatchNorm1d(hid_dim) if use_batchnorm else nn.Identity())
            self.hidden_lns.append(nn.LayerNorm(hid_dim) if use_layernorm else nn.Identity())
        
        # Output layer
        self.output_layer = nn.Linear(hid_dim, output_dim)

        print(f"""Model input dim: {input_dim},
            hidden dim: {hid_dim},
            output dim: {output_dim},
            num layers: {num_layers},
            activation: {activation},
            dropout: {dropout},
            use_batchnorm: {use_batchnorm},
            use_layernorm: {use_layernorm}""")

        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "elu":
            self.activation_fn = nn.ELU()
        elif activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(negative_slope=0.1)

        self.reset_parameters()

    def reset_parameters(self, init_type: str = "xavier_uniform"):
        """
        Initializes model parameters with a variety of initialization methods.
        Supported types: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal', 'normal', 'uniform'
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(layer.weight)
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity=self.activation)
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity=self.activation)
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(layer.weight)
                elif init_type == "normal":
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                elif init_type == "uniform":
                    nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
                else:
                    raise ValueError(f"Unknown init_type: {init_type}")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        
    def forward(self, x):
        # print(f"Initial x: {x}")
        x = self.dropout(x)
        # Input layer with normalization
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.input_ln(x)
        x = self.activation_fn(x)
        # Residual connections in hidden layers
        for hidden_layer, hidden_bn, hidden_ln in zip(self.hidden_layers, self.hidden_bns, self.hidden_lns):
            residual = x
            x = hidden_layer(x)
            x = hidden_bn(x)
            x = hidden_ln(x)
            x = self.activation_fn(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection
        # Output layer
        output = self.output_layer(x)
        return output
