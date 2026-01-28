import math
from typing import Union, Tuple, Optional

import torch
from torch.func import stack_module_state, functional_call, vmap
# Assuming these imports are in your project structure
from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.Util import get_encoder
from FunctionEncoder.Model.Architecture.MLP import get_activation, MLP



def rk4_difference_only(model, xs, dt):
    """
    Runge-Kutta 4th order method for solving ODEs.
    This method correctly computes the change in state.
    """
    k1 = model(xs)
    k2 = model(xs + dt / 2 * k1)
    k3 = model(xs + dt / 2 * k2)
    k4 = model(xs + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class BA_NODE(BaseArchitecture):
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size=77, n_layers=4, ode_state_size=77, learn_basis_functions=True, *args, **kwargs):
        """Modified to account for encoder and decoder."""
        input_dim = input_size[0]
        output_dim = output_size[0]
        
        # Encoder params
        n_params = (input_dim + 1) * ode_state_size
        
        # Dynamics model params (input=output=ode_state_size)
        dynamics_params = (ode_state_size + 1) * hidden_size + \
                          (hidden_size + 1) * hidden_size * (n_layers - 2) + \
                          (hidden_size + 1) * ode_state_size
        
        if learn_basis_functions:
            n_params += dynamics_params * n_basis
        else:
            n_params += dynamics_params
            
        # Decoder params
        n_params += (ode_state_size + 1) * output_dim
        
        return n_params

    def __init__(self,
                 input_size: Tuple[int],
                 output_size: Tuple[int],
                 n_basis: int = 100,
                 hidden_size: int = 77,
                 # New parameter for the internal dimension of the ODE
                 ode_state_size: int = 64,
                 n_layers: int = 4,
                 activation: str = "silu",
                 learn_basis_functions=True,
                 dt: float = 0.1,
                 prediction_length: int = 1, # Add prediction length
                 encoder_type: Optional[str] = None,
                 encoder_kwargs: dict = {},
                 **kwargs):
        super(BA_NODE, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        
        # Determine feature dimension from input_size
        feature_dim = input_size[-1]
        
        if encoder_type is not None:
            self.history_encoder = get_encoder(encoder_type, input_size=feature_dim, **encoder_kwargs)
            encoder_input_size = (feature_dim,)
        else:
            self.history_encoder = None
            encoder_input_size = input_size

        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.learn_basis_functions = learn_basis_functions
        self.dt = dt
        self.prediction_length = prediction_length

        if not self.learn_basis_functions:
            n_basis = 1
            self.n_basis = 1

        # Encoder maps from input_size to the internal ODE state size
        self.encoder = MLP(
            input_size=encoder_input_size, 
            output_size=(ode_state_size,), 
            n_basis=1, 
            hidden_size=hidden_size, 
            n_layers=2,
            activation=activation, 
            learn_basis_functions=False
        )

        # The core dynamics models now operate on the internal state size
        self.dynamics_models = torch.nn.ModuleList([
            MLP(
              input_size=(ode_state_size,), # Input is the internal state
              output_size=(ode_state_size,),# Output is also the internal state
              n_basis=1,
              hidden_size=hidden_size,
              n_layers=n_layers,
              activation=activation,
              learn_basis_functions=False,
            )
            for _ in range(n_basis)
        ])

        self._dyn_params, self._dyn_buffers = stack_module_state(self.dynamics_models)
        
        self.project_basis_functions = torch.nn.Linear(n_basis, 1)
        # Decoder maps from the internal ODE state size back to the desired output_size (per step)
        # If prediction_length > 1, output_size is flattened (P*D,). We need D.
        if self.prediction_length > 1:
            assert output_size[0] % self.prediction_length == 0, f"Output size {output_size[0]} must be divisible by prediction length {self.prediction_length}"
            per_step_output_dim = output_size[0] // self.prediction_length
            decoder_output_size = (per_step_output_dim,)
        else:
            decoder_output_size = output_size

        self.decoder = MLP(
            input_size=(ode_state_size,), 
            output_size=decoder_output_size, 
            n_basis=1, 
            hidden_size=hidden_size,
            n_layers=2,
            activation=activation, 
            learn_basis_functions=False
        )

    def _single_model_rk4(self, params, buffers, h):
        # This function runs ONE model, but we will vmap it later
        
        # Define the function f(y) for this specific set of params
        def f(y):
            # functional_call(module, (params, buffers), args)
            return functional_call(self.dynamics_models[0], (params, buffers), (y,))
            
        # Solve RK4 for this single model
        # (Assumes rk4_difference_only is available in scope)
        return rk4_difference_only(f, h, self.dt)

    def _ensemble_delta(self, h):
        # h: (..., S)
        # vmap over models (params/buffers dim 0), share h
        params, buffers = stack_module_state(self.dynamics_models)
        delta_h = vmap(
            self._single_model_rk4,
            in_dims=(0, 0, None),
        )(params, buffers, h)
        # delta_h: (K, ..., S)
        delta_h = delta_h.movedim(0, -1)  # (..., S, K) to match your code
        return delta_h

    @torch.compile(mode="reduce-overhead")
    def _rollout(self, h: torch.Tensor, prediction_horizon: int):
        """
        h: (..., S)   initial latent state from encoder
        returns:
            h: final latent state (..., S)
            Gs_seq: (..., P, O, K)  where
                P = prediction_horizon
                O = per-step output dim
                K = n_basis
        """
        # Static dims
        batch_dims = h.shape[:-1]             # (...,)
        ode_dim    = h.shape[-1]              # S
        num_bases  = self.n_basis             # K
        output_dim = self.decoder.output_size[0]  # O

        # Preallocate output: (..., P, O, K)
        Gs_seq = h.new_empty(*batch_dims,
                             prediction_horizon,
                             output_dim,
                             num_bases)

        # Rollout over time
        for t in range(prediction_horizon):
            # Ensemble dynamics: (..., S, K)
            delta_h_stacked = self._ensemble_delta(h)

            # ODE update + basis projection
            h_expanded = h.unsqueeze(-1)              # (..., S, 1)
            h_next     = h_expanded + delta_h_stacked # (..., S, K)
            h          = self.project_basis_functions(h_next).squeeze(-1)  # (..., S)

            # Decode all bases at this step
            # h_next: (..., S, K)
            # reshape to (B*K, S)
            h_next_reshaped = (
                h_next.movedim(-2, -1)    # (..., K, S)
                      .reshape(-1, ode_dim)
            )
            decoded_reshaped = self.decoder(h_next_reshaped)  # (B*K, O)

            # Back to (..., O, K)
            Gs_step = (
                decoded_reshaped
                .view(*batch_dims, num_bases, output_dim)  # (..., K, O)
                .movedim(-2, -1)                           # (..., O, K)
            )

            # Write into preallocated tensor at time t
            Gs_seq[..., t, :, :] = Gs_step

        return h, Gs_seq

    def forward(self, x, prediction_horizon: int = 1):
        # Check against the last dimension (features)
        assert x.shape[-1] == self.input_size[-1], f"Expected input size {self.input_size[-1]}, got {x.shape[-1]}"
        
        if self.history_encoder is not None:
            if len(x.shape) == 4:
                original_shape = x.shape
                batch_size, history_length, feature_dim = original_shape[0] * original_shape[1], original_shape[2], original_shape[3]
                x_reshaped = x.reshape(batch_size, history_length, feature_dim)
                x = self.history_encoder(x_reshaped).reshape(original_shape[0], original_shape[1], -1)
            else:
                x = self.history_encoder(x)

        reshape = None
        if len(x.shape) == 1:
            reshape = 1
            x = x.reshape(1, 1, -1)
        if len(x.shape) == 2:
            reshape = 2
            x = x.unsqueeze(0)

        # Encode the input to the initial hidden state for the ODE
        h0 = self.encoder(x) # h0
        _, Gs_seq = self._rollout(h0, prediction_horizon)
        
        Gs_flattened = Gs_seq.flatten(-3, -2)  # merge P and O
                        
        if not self.learn_basis_functions:
            Gs = Gs_flattened.view(*x.shape[:2], *self.output_size)
        else:
            Gs = Gs_flattened

        # send back to the given shape
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        if reshape == 2:
            Gs = Gs.squeeze(0)
        return Gs