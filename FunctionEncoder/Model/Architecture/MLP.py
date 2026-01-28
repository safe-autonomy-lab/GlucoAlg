import torch
from typing import Optional, Tuple
from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.Util import get_encoder
from FunctionEncoder.Model.Architecture.ITransformer import VariateEmbedding, FeedForward, ITBlock, ITransformer

# Returns the desired activation function by name
def get_activation(activation):
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == 'gelu':
        return torch.nn.GELU()
    elif activation == 'silu' or activation == 'swish': # Add this block
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Unknown activation: {activation}")

class MLP(BaseArchitecture):
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size=256, n_layers=4, learn_basis_functions=True, *args, **kwargs):
        input_size = input_size[0]
        output_size = output_size[0] * n_basis if learn_basis_functions else output_size[0]
        n_params =  input_size * hidden_size + hidden_size + \
                    (n_layers - 2) * hidden_size * hidden_size + (n_layers - 2) * hidden_size + \
                    hidden_size * output_size + output_size
        return n_params

    def __init__(self,
                 input_size:Tuple[int],
                 output_size:Tuple[int],
                 n_basis:int=100,
                 hidden_size:int=256,
                 n_layers:int=4,
                 activation:str="relu",
                 learn_basis_functions=True,
                 encoder_type: Optional[str] = None,
                 encoder_kwargs: dict = {},
                 **kwargs):
        super(MLP, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        assert encoder_type is None or encoder_type in ["attention", "attn", "lstm", "itransformer"], "encoder_type must be None or one of ['attention', 'attn', 'lstm', 'itransformer']"
        assert encoder_type is None or isinstance(encoder_kwargs, dict), "encoder_kwargs must be provided"

        
        self.history_encoder = get_encoder(encoder_type, **encoder_kwargs) if encoder_type is not None else None
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.learn_basis_functions = learn_basis_functions

        # Input size depends on whether we are using a history encoder
        # self.input_size = (input_size[0], ) if self.history_encoder is None else (self.history_encoder.hidden_size, )
        self.input_size = (input_size[-1], )
            
        output_size = output_size[0] * n_basis if learn_basis_functions else output_size[0]
        # build net
        layers = []
        layers.append(torch.nn.Linear(self.input_size[0], hidden_size))
        
        layers.append(get_activation(activation))
        for _ in range(n_layers - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation(activation))
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.model = torch.nn.Sequential(*layers)

        # verify number of parameters        
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, n_layers, learn_basis_functions=learn_basis_functions)
        if self.history_encoder is not None:
            estimated_n_params += sum([p.numel() for p in self.history_encoder.parameters()])

        assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."


    def forward(self, x):
        original_shape = x.shape
    
        if self.history_encoder is not None:
            # Handle temporal data: x shape is (n_functions, n_examples, history_length, feature_dim)
            if len(x.shape) == 4:
                # Reshape from 4D to 3D for history encoder
                # (n_functions, n_examples, history_length, feature_dim) -> (batch_size, history_length, feature_dim)
                batch_size = x.shape[0] * x.shape[1]  # n_functions * n_examples
                history_length = x.shape[2]
                feature_dim = x.shape[3]

                # Reshape to 3D for history encoder
                x_reshaped = x.view(batch_size, history_length, feature_dim)
                x_encoded = self.history_encoder(x_reshaped)  # (batch_size, hidden_size)
                # Reshape back to (n_functions, n_examples, hidden_size)
                x = x_encoded.view(original_shape[0], original_shape[1], -1)
                
            else:
                # For non-4D tensors, use history encoder directly
                x = self.history_encoder(x)
        
        # Update input size check for encoded features
        if self.history_encoder is not None:
            # After encoding, x should have shape (n_functions, n_examples, hidden_size)
            # The hidden_size becomes our effective input_size
            expected_input_size = x.shape[-1]
        else:
            expected_input_size = self.input_size[0]
            assert x.shape[-1] == expected_input_size, f"Expected input size {expected_input_size}, got {x.shape[-1]}"
        
        # Handle different input shapes for the MLP
        reshape = None
        if len(x.shape) == 1:
            reshape = 1
            x = x.reshape(1, 1, -1)
        elif len(x.shape) == 2:
            reshape = 2
            x = x.unsqueeze(0)
        # x.shape is now (n_functions, n_examples, feature_size)

        # Main MLP forward pass
        outs = self.model(x)
        
        # Reshape outputs according to basis functions
        if self.learn_basis_functions:
            Gs = outs.view(*x.shape[:2], *self.output_size, self.n_basis)
        else:
            Gs = outs.view(*x.shape[:2], *self.output_size)

        # Restore original shape if needed
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        elif reshape == 2:
            Gs = Gs.squeeze(0)
            
        return Gs


