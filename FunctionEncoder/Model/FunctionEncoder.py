from typing import Union, Tuple, Dict, Any
import logging
import torch
import gc
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.Euclidean import Euclidean
from FunctionEncoder.Model.Architecture.MLP import MLP
from FunctionEncoder.Model.Architecture.BA_NODE import BA_NODE
from FunctionEncoder.Model.Architecture.NeuralODE import NeuralODE
from FunctionEncoder.Model.Architecture.ParallelMLP import ParallelMLP
from FunctionEncoder.Model.Architecture.ITransformer import ITransformer


class FunctionEncoder(torch.nn.Module):
    """A function encoder learns basis functions/vectors over a Hilbert space.

    A function encoder learns basis functions/vectors over a Hilbert space. 
    Typically, this is a function space mapping to Euclidean vectors, but it can be any Hilbert space, IE probability distributions.
    This class has a general purpose algorithm which supports both deterministic and stochastic data.
    The only difference between them is the dataset used and the inner product definition.
    This class supports two methods for computing the coefficients of the basis function, also called a representation:
    1. "inner_product": It computes the inner product of the basis functions with the data via a Monte Carlo approximation.
    2. "least_squares": This method computes the least squares solution in terms of vector operations. This typically trains faster and better. 
    This class also supports the residuals method, which learns the average function in the dataset. The residuals/error of this approximation, 
    for each function in the space, is learned via a function encoder. This tends to improve performance when the average function is not f(x) = 0. 
    """

    def __init__(self,
                 input_size:Tuple[int], 
                 output_size:Tuple[int], 
                 data_type:str, 
                 n_basis:int=100, 
                 model_type:Union[str, type]="MLP",
                 model_kwargs:Dict[str, Any]=dict(),
                 method:str="least_squares", 
                 use_residuals_method:bool=False,  
                 regularization_parameter:float=1.0, # if you normalize your data, this is usually good
                 gradient_accumulation:int=1, # default: no gradient accumulation
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs:Dict[str, Any]={"lr":1e-3},
                 device:str="cuda" if torch.cuda.is_available() else "cpu"
                 ):
        """ Initializes a function encoder.

        Args:
        input_size: tuple[int]: The size of the input space, e.g. (1,) for 1D input
        output_size: tuple[int]: The size of the output space, e.g. (1,) for 1D output
        data_type: str: "deterministic" or "stochastic". Determines which defintion of inner product is used.
        n_basis: int: Number of basis functions to use.
        model_type: str: The type of model to use. See the types and kwargs in FunctionEncoder/Model/Architecture. Typically a MLP.
        model_kwargs: Union[dict, type(None)]: The kwargs to pass to the model. See the types and kwargs in FunctionEncoder/Model/Architecture.
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        use_residuals_method: bool: Whether to use the residuals method. If True, uses an average function to predict the average of the data, and then learns the error with a function encoder.
        regularization_parameter: float: The regularization parameter for the least squares method, that encourages the basis functions to be unit length. 1 is usually good, but if your ys are very large, this may need to be increased.
        gradient_accumulation: int: The number of batches to accumulate gradients over. Typically its best to have n_functions>=10 or so, and have gradient_accumulation=1. However, sometimes due to memory reasons, or because the functions do not have the same amount of data, its necesary for n_functions=1 and gradient_accumulation>=10.
        """
        if model_type == "MLP":
            assert len(input_size) <= 2, "MLP only supports 1D or 2D input, but 2D input is required for the history encoder"
        if model_type == "ParallelMLP":
            assert len(input_size) == 1, "ParallelMLP only supports 1D input"
        if model_type == "CNN":
            assert len(input_size) == 3, "CNN only supports 3D input"
        if isinstance(model_type, type):
            assert issubclass(model_type, BaseArchitecture), "model_type should be a subclass of BaseArchitecture. This just gives a way of predicting the number of parameters before init."
        assert len(input_size) in [1, 2, 3], "Input must either be 1-Dimensional (euclidean vector), 2-Dimensional (history encoder) or 3-Dimensional (image)"
        assert input_size[0] >= 1, "Input size must be at least 1"
        # assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic", "categorical"], f"Unknown data type: {data_type}"
        super(FunctionEncoder, self).__init__()
        
        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.method = method
        self.data_type = data_type
        self.device = device
        
        # models and optimizers
        self.model = self._build(model_type, model_kwargs).to(self.device)
        self.average_function = self._build(model_type, model_kwargs, average_function=True).to(self.device) if use_residuals_method else None
        params = [*self.model.parameters()]
        if self.average_function is not None:
            params += [*self.average_function.parameters()]
            if self.average_function.history_encoder is not None:
                params += [*self.average_function.history_encoder.parameters()]
        self.opt = optimizer(params, **optimizer_kwargs) # usually ADAM with lr 1e-3

        # regulation only used for LS method
        self.regularization_parameter = regularization_parameter
        # accumulates gradients over multiple batches, typically used when n_functions=1 for memory reasons. 
        self.gradient_accumulation = gradient_accumulation

        # for printing
        self.model_type = model_type
        self.model_kwargs = model_kwargs

        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        if self.model.history_encoder is not None:
            input_size = (input_size[-1], )
            # input_size = (self.model.history_encoder.hidden_size, )

        estimated_n_params = FunctionEncoder.predict_number_params(input_size=input_size, output_size=output_size, n_basis=n_basis, model_type=model_type, model_kwargs=model_kwargs, use_residuals_method=use_residuals_method)
        
        if self.model.history_encoder is not None:
            estimated_n_params += sum([p.numel() for p in self.model.history_encoder.parameters()])
        if self.average_function is not None and self.average_function.history_encoder is not None:
            estimated_n_params += sum([p.numel() for p in self.average_function.history_encoder.parameters()])
        # assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."



    def _build(self, 
               model_type:Union[str, type],
               model_kwargs:Dict[str, Any], 
               average_function:bool=False) -> torch.nn.Module:
        """Builds a function encoder as a single model. Can also build the average function. 
        
        Args:
        model_type: Union[str, type]: The type of model to use. See the types and kwargs in FunctionEncoder/Model/Architecture. Typically "MLP", can also be a custom class.
        model_kwargs: Dict[str, Any]: The kwargs to pass to the model. See the kwargs in FunctionEncoder/Model/Architecture/.
        average_function: bool: Whether to build the average function. If True, builds a single function model.

        Returns:
        torch.nn.Module: The basis functions or average function model.
        """

        # if provided as a string, parse the string into a class
        if type(model_type) == str:
            if model_type == "MLP":
                return MLP(input_size=self.input_size,
                           output_size=self.output_size,
                           n_basis=self.n_basis,
                           learn_basis_functions=not average_function,
                           **model_kwargs)
            if model_type == "ParallelMLP":
                return ParallelMLP(input_size=self.input_size,
                                   output_size=self.output_size,
                                   n_basis=self.n_basis,
                                   learn_basis_functions=not average_function,
                                   **model_kwargs)
            elif model_type == "Euclidean":
                return Euclidean(input_size=self.input_size,
                                 output_size=self.output_size,
                                 n_basis=self.n_basis,
                                 **model_kwargs)
            elif model_type == "CNN":
                return CNN(input_size=self.input_size,
                           output_size=self.output_size,
                           n_basis=self.n_basis,
                           learn_basis_functions=not average_function,
                           **model_kwargs)
            elif model_type == "NeuralODE":
                return NeuralODE(input_size=self.input_size,
                                    output_size=self.output_size,
                                    n_basis=self.n_basis,
                                    learn_basis_functions=not average_function,
                                    **model_kwargs)
            elif model_type == "BA_NODE":
                return BA_NODE(input_size=self.input_size,
                                    output_size=self.output_size,
                                    n_basis=self.n_basis,
                                    learn_basis_functions=not average_function,
                                    **model_kwargs)
            elif model_type == "ITransformer":
                return ITransformer(input_len=self.input_size[0],
                                    pred_len=self.output_size[0],
                                    n_variates=self.input_size[1],
                                    **model_kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}. Should be one of 'MLP', 'ParallelMLP', 'Euclidean', 'CNN', 'NeuralODE', or 'ITransformer'")
        else:  # otherwise, assume it is a class and directly instantiate it
            return model_type(input_size=self.input_size,
                              output_size=self.output_size,
                              n_basis=self.n_basis,
                              learn_basis_functions=not average_function,
                              **model_kwargs)

    def compute_representation(self, 
                               example_xs:torch.tensor, 
                               example_ys:torch.tensor, 
                               method:str="least_squares", 
                               prediction_horizon:int=1,
                               **kwargs: Any) -> Tuple[torch.tensor, Union[torch.tensor, None]]:
        """Computes the coefficients of the basis functions.

        This method does the forward pass of the basis functions (and the average function if it exists) over the example data.
        Then it computes the coefficients of the basis functions via a Monte Carlo integration of the inner product with the example data.
        
        Args:
        example_xs: torch.tensor: The input data. Shape (n_example_datapoints, input_size) or (n_functions, n_example_datapoints, input_size)
        example_ys: torch.tensor: The output data. Shape (n_example_datapoints, output_size) or (n_functions, n_example_datapoints, output_size)
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        kwargs: Any: Additional kwargs to pass to the least squares method.

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis) or (n_basis,) if n_functions=1. 
        Union[torch.tensor, None]: The gram matrix if using least squares method. None otherwise.
        """
        
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"example_xs must have shape (..., {self.input_size}). Expected {self.input_size}, got {example_xs.shape[-len(self.input_size):]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"example_ys must have shape (..., {self.output_size}). Expected {self.output_size}, got {example_ys.shape[-len(self.output_size):]}"
        assert example_xs.shape[:-len(self.input_size)] == example_ys.shape[:-len(self.output_size)], f"example_xs and example_ys must have the same shape except for the last {len(self.input_size)} dimensions. Expected {example_xs.shape[:-len(self.input_size)]}, got {example_ys.shape[:-len(self.output_size)]}"

        # if not in terms of functions, add a function batch dimension
        reshaped = False
        if len(example_xs.shape) - len(self.input_size) == 1:
            reshaped = True
            example_xs = example_xs.unsqueeze(0)
            example_ys = example_ys.unsqueeze(0)

        # optionally subtract average function if we are using residuals method
        # we dont want to backprop to the average function. So we block grads. 
        if self.average_function is not None:
            with torch.no_grad():
                example_y_hat_average = self.average_function.forward(example_xs)
                example_ys = example_ys - example_y_hat_average

        # compute representation
        Gs = self.model.forward(example_xs, prediction_horizon=prediction_horizon) # forward pass of the basis functions
        if method == "inner_product":
            representation = self._compute_inner_product_representation(Gs, example_ys)
            gram = None
        elif method == "least_squares":
            representation, gram = self._compute_least_squares_representation(Gs, example_ys, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # reshape if necessary
        if reshaped:
            assert representation.shape[0] == 1, "Expected a single function batch dimension"
            representation = representation.squeeze(0)
        return representation, gram

    def _deterministic_inner_product(self, 
                                     fs:torch.tensor, 
                                     gs:torch.tensor,) -> torch.tensor:
        """Approximates the L2 inner product between fs and gs using a Monte Carlo approximation.
        Latex: \langle f, g \rangle = \frac{1}{V}\int_X f(x)g(x) dx \approx \frac{1}{n} \sum_{i=1}^n f(x_i)g(x_i)
        Note we are scaling the L2 inner product by 1/volume, which removes volume from the monte carlo approximation.
        Since scaling an inner product is still a valid inner product, this is still an inner product.
        
        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True

        # compute inner products via MC integration
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _stochastic_inner_product(self, 
                                  fs:torch.tensor, 
                                  gs:torch.tensor,) -> torch.tensor:
        """ Approximates the logit version of the inner product between continuous distributions. 
        Latex: \langle f, g \rangle = \int_X (f(x) - \Bar{f}(x) )(g(x) - \Bar{g}(x)) dx \approx \frac{1}{n} \sum_{i=1}^n (f(x_i) - \Bar{f}(x_i))(g(x_i) - \Bar{g}(x_i))
        
        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2] == 1, f"Expected fs and gs to have the same output size, which is 1 for the stochastic case since it learns the pdf(x), got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=1, keepdim=True)
        mean_g = torch.mean(gs, dim=1, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)
        # Technically we should multiply by volume, but we are assuming that the volume is 1 since it is often not known

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _categorical_inner_product(self,
                                   fs:torch.tensor,
                                   gs:torch.tensor,) -> torch.tensor:
        """ Approximates the inner product between discrete conditional probability distributions.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3, 4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3, 4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, which is the number of categories in this case, got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=2, keepdim=True)
        mean_g = torch.mean(gs, dim=2, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _inner_product(self, 
                       fs:torch.tensor, 
                       gs:torch.tensor) -> torch.tensor:
        """ Computes the inner product between fs and gs. This passes the data to either the deterministic or stochastic inner product methods.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, got {fs.shape[2]} and {gs.shape[2]}"

        if self.data_type == "deterministic":
            return self._deterministic_inner_product(fs, gs)
        elif self.data_type == "stochastic":
            return self._stochastic_inner_product(fs, gs)
        elif self.data_type == "categorical":
            return self._categorical_inner_product(fs, gs)
        else:
            raise ValueError(f"Unknown data type: '{self.data_type}'. Should be 'deterministic', 'stochastic', or 'categorical'")

    def _norm(self, fs:torch.tensor, squared=False) -> torch.tensor:
        """ Computes the norm of fs according to the chosen inner product.

        Args:
        fs: torch.tensor: The function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)

        Returns:
        torch.tensor: The Hilbert norm of fs.
        """
        norm_squared = self._inner_product(fs, fs)
        if not squared:
            return norm_squared.sqrt()
        else:
            return norm_squared

    def _distance(self, fs:torch.tensor, gs:torch.tensor, squared=False) -> torch.tensor:
        """ Computes the distance between fs and gs according to the chosen inner product.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        gs: torch.tensor: The second set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        returns:
        torch.tensor: The distance between fs and gs.
        """
        return self._norm(fs - gs, squared=squared)

    def _compute_inner_product_representation(self, 
                                              Gs:torch.tensor, 
                                              example_ys:torch.tensor) -> torch.tensor:
        """ Computes the coefficients via the inner product method.

        Args:
        Gs: torch.tensor: The basis functions. Shape (n_functions, n_datapoints, output_size, n_basis)
        example_ys: torch.tensor: The output data. Shape (n_functions, n_datapoints, output_size)

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        """
        
        assert len(Gs.shape)== 4, f"Expected Gs to have shape (f,d,m,k), got {Gs.shape}"
        assert len(example_ys.shape) == 3, f"Expected example_ys to have shape (f,d,m), got {example_ys.shape}"
        assert Gs.shape[0] == example_ys.shape[0], f"Expected Gs and example_ys to have the same number of functions, got {Gs.shape[0]} and {example_ys.shape[0]}"
        assert Gs.shape[1] == example_ys.shape[1], f"Expected Gs and example_ys to have the same number of datapoints, got {Gs.shape[1]} and {example_ys.shape[1]}"
        assert Gs.shape[2] == example_ys.shape[2], f"Expected Gs and example_ys to have the same output size, got {Gs.shape[2]} and {example_ys.shape[2]}"

        # take inner product with Gs, example_ys
        inner_products = self._inner_product(Gs, example_ys)
        return inner_products

    def _compute_least_squares_representation(self, 
                                              Gs:torch.tensor, 
                                              example_ys:torch.tensor, 
                                              lambd:Union[float, None]= None) -> Tuple[torch.tensor, torch.tensor]:
        """ Computes the coefficients via the least squares method.
        
        Args:
        Gs: torch.tensor: The basis functions. Shape (n_functions, n_datapoints, output_size, n_basis)
        example_ys: torch.tensor: The output data. Shape (n_functions, n_datapoints, output_size)
        lambd: float: The regularization parameter. None by default. If None, scales with 1/n_datapoints.
        
        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        torch.tensor: The gram matrix. Shape (n_functions, n_basis, n_basis)
        """
        
        # Check if Gs is 3D (missing basis dim or output dim collapsed?)
        # If Gs is (B, T, D) and example_ys is (B, T, D), we might be in a direct prediction mode
        # But this method expects basis functions.
        # Let's adhere to 4D Gs: (f, d, m, k).
        
        if len(Gs.shape) == 3:
             # Assume n_basis=1 and unsqueeze? Or output_size?
             # Gs: (f, d, m) -> (f, d, m, 1)
             Gs = Gs.unsqueeze(-1)
        
        assert len(Gs.shape)== 4, f"Expected Gs to have shape (f,d,m,k), got {Gs.shape}"
        assert len(example_ys.shape) == 3, f"Expected example_ys to have shape (f,d,m), got {example_ys.shape}"
        assert Gs.shape[0] == example_ys.shape[0], f"Expected Gs and example_ys to have the same number of functions, got {Gs.shape[0]} and {example_ys.shape[0]}"
        assert Gs.shape[1] == example_ys.shape[1], f"Expected Gs and example_ys to have the same number of datapoints, got {Gs.shape[1]} and {example_ys.shape[1]}"
        # Relax output size check or ensure it matches
        # assert Gs.shape[2] == example_ys.shape[2], f"Expected Gs and example_ys to have the same output size, got {Gs.shape[2]} and {example_ys.shape[2]}"
        assert lambd is None or lambd >= 0, f"Expected lambda to be non-negative or None, got {lambd}"

        # Handle 0 datapoints (no context)
        if Gs.shape[1] == 0:
            # Return zeros
            n_functions = Gs.shape[0]
            n_basis = Gs.shape[3]
            device = Gs.device
            return torch.zeros((n_functions, n_basis), device=device), torch.eye(n_basis, device=device).unsqueeze(0).repeat(n_functions, 1, 1)

        # set lambd to decrease with more data
        if lambd is None:
            lambd = 1e-3 # emprically this does well. We need to investigate if there is an optimal value here.

        # compute gram
        gram = self._inner_product(Gs, Gs)
        gram_reg = gram + lambd * torch.eye(self.n_basis, device=gram.device)

        # compute the matrix G^TF
        ip_representation = self._inner_product(Gs, example_ys)

        # Compute (G^TG)^-1 G^TF
        ls_representation = torch.einsum("fkl,fl->fk", gram_reg.inverse(), ip_representation) # this is just batch matrix multiplication
        return ls_representation, gram

    def predict(self, 
                query_xs:torch.tensor,
                representations:torch.tensor, 
                precomputed_average_ys:Union[torch.tensor, None]=None,
                prediction_horizon:int=1,
                **kwargs: Any
                ) -> torch.tensor:
        """ Predicts the output of the function encoder given the input data and the coefficients of the basis functions. Uses the average function if it exists.

        Args:
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        representations: torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        precomputed_average_ys: Union[torch.tensor, None]: The average function output. If None, computes it. Shape (n_functions, n_datapoints, output_size)
        prediction_horizon: int: The number of steps to predict.
        kwargs: dict: Additional kwargs to pass to the predict method.
        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """

        assert len(query_xs.shape) == 2 + len(self.input_size), f"Expected xs to have shape (f,d,*n), got {query_xs.shape}"
        assert len(representations.shape) == 2, f"Expected representations to have shape (f,k), got {representations.shape}"
        assert query_xs.shape[0] == representations.shape[0], f"Expected xs and representations to have the same number of functions, got {query_xs.shape[0]} and {representations.shape[0]}"

        # this is weighted combination of basis functions
        Gs = self.model.forward(query_xs, prediction_horizon=prediction_horizon)
        
        if len(Gs.shape) == 3:
             Gs = Gs.unsqueeze(-1)

        y_hats = torch.einsum("fdmk,fk->fdm", Gs, representations)
        
        # optionally add the average function
        # it is allowed to be precomputed, which is helpful for training
        # otherwise, compute it
        if self.average_function:
            if precomputed_average_ys is not None:
                average_ys = precomputed_average_ys
            else:
                average_ys = self.average_function.forward(query_xs)
            y_hats = y_hats + average_ys
        return y_hats

    def predict_from_examples(self, 
                              example_xs:torch.tensor, 
                              example_ys:torch.tensor, 
                              query_xs:torch.tensor,
                              method:str="least_squares",
                              **kwargs: Any):
        """ Predicts the output of the function encoder given the input data and the example data. Uses the average function if it exists.
        
        Args:
        example_xs: torch.tensor: The example input data used to compute a representation. Shape (n_example_datapoints, input_size)
        example_ys: torch.tensor: The example output data used to compute a representation. Shape (n_example_datapoints, output_size)
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        kwargs: dict: Additional kwargs to pass to the least squares method.

        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """

        assert len(example_xs.shape) == 2 + len(self.input_size), f"Expected example_xs to have shape (f,d,*n), got {example_xs.shape}"
        assert len(example_ys.shape) == 2 + len(self.output_size), f"Expected example_ys to have shape (f,d,*m), got {example_ys.shape}"
        assert len(query_xs.shape) == 2 + len(self.input_size), f"Expected xs to have shape (f,d,*n), got {query_xs.shape}"
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"Expected example_xs to have shape (..., {self.input_size}), got {example_xs.shape[-1]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"Expected example_ys to have shape (..., {self.output_size}), got {example_ys.shape[-1]}"
        assert query_xs.shape[-len(self.input_size):] == self.input_size, f"Expected xs to have shape (..., {self.input_size}), got {query_xs.shape[-1]}"
        assert example_xs.shape[0] == example_ys.shape[0], f"Expected example_xs and example_ys to have the same number of functions, got {example_xs.shape[0]} and {example_ys.shape[0]}"
        assert example_xs.shape[1] == example_xs.shape[1], f"Expected example_xs and example_ys to have the same number of datapoints, got {example_xs.shape[1]} and {example_ys.shape[1]}"
        assert example_xs.shape[0] == query_xs.shape[0], f"Expected example_xs and xs to have the same number of functions, got {example_xs.shape[0]} and {query_xs.shape[0]}"

        representations, _ = self.compute_representation(example_xs, example_ys, method=method, **kwargs)
        y_hats = self.predict(query_xs, representations)
        return y_hats


    def estimate_L2_error(self, example_xs, example_ys):
        """ Estimates the L2 error of the function encoder on the example data. 
        This gives an idea if the example data lies in the span of the basis, or not.
        
        Args:
        example_xs: torch.tensor: The example input data used to compute a representation. Shape (n_functions, n_example_datapoints, input_size)
        example_ys: torch.tensor: The example output data used to compute a representation. Shape (n_functions, n_example_datapoints, output_size)
        
        Returns:
        torch.tensor: The estimated L2 error. Shape (n_functions,)
        """
        representation, gram = self.compute_representation(example_xs, example_ys, method="least_squares")
        f_hat_norm_squared = representation @ gram @ representation.T
        f_norm_squared = self._inner_product(example_ys, example_ys)
        l2_distance = torch.sqrt(f_norm_squared - f_hat_norm_squared)
        return l2_distance



    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    batch_size: int = 32,
                    progress_bar=True,
                    callback:BaseCallback=None,
                    **kwargs: Any):
        """ Trains the function encoder on the dataset for some number of epochs.
        
        Args:
        dataset: BaseDataset: The dataset to train on.
        epochs: int: The number of epochs to train for.
        batch_size: int: Number of functions to process per batch. If None, uses dataset.n_functions.
        progress_bar: bool: Whether to show a progress bar.
        callback: BaseCallback: A callback to use during training. Can be used to test loss, etc. 
        
        Returns:
        list[float]: The losses at each epoch."""

        # verify dataset is correct
        dataset.check_dataset()
        
        # set device
        device = next(self.parameters()).device
        
        # Let callbacks know starting data
        if callback is not None:
            callback.on_training_start(locals())

        # method to use for representation during training
        assert self.method in ["inner_product", "least_squares"], f"Unknown method: {self.method}"

        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        
        current_batch_size = batch_size
        
        for epoch in bar:
            # Reset batch state for new epoch if supported
            if hasattr(dataset, 'reset_batch_state'):
                dataset.reset_batch_state()
            
            # Store original n_functions to restore later
            original_n_functions = dataset.n_functions
            
            # Calculate number of batches per epoch
            if hasattr(dataset, 'train_X'):
                total_functions = dataset.train_X.shape[0]
                n_batches = max(1, (total_functions + current_batch_size - 1) // current_batch_size)
            else:
                # Fallback for datasets without train_X attribute
                n_batches = max(1, (dataset.n_functions + current_batch_size - 1) // current_batch_size)
            
            epoch_loss = 0.0
            
            # Process multiple batches per epoch
            for batch_idx in range(n_batches):
                # Set batch size for this iteration
                dataset.n_functions = min(current_batch_size, 
                                        getattr(dataset, 'train_X', torch.zeros(dataset.n_functions)).shape[0] - batch_idx * current_batch_size)
                
                if dataset.n_functions <= 0:
                    break
                    
                example_xs, example_ys, query_xs, query_ys, _ = dataset.sample()

                # train average function, if it exists
                if self.average_function is not None:
                    # predict averages
                    expected_yhats = self.average_function.forward(query_xs).to(device)

                    # compute average function loss
                    average_function_loss = self._distance(expected_yhats, query_ys, squared=True).mean()
                    
                    # we only train average function to fit data in general, so block backprop from the basis function loss
                    expected_yhats = expected_yhats.detach()
                else:
                    expected_yhats = None

                # approximate functions, compute error
                representation, gram = self.compute_representation(example_xs, example_ys, method=self.method, **kwargs)
                y_hats = self.predict(query_xs, representation, precomputed_average_ys=expected_yhats)
                prediction_loss = self._distance(y_hats, query_ys, squared=True).mean()

                # LS requires regularization since it does not care about the scale of basis
                # so we force basis to move towards unit norm. They dont actually need to be unit, but this prevents them
                # from going to infinity.
                if self.method == "least_squares":
                    norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1)**2).mean()

                # add loss components
                batch_loss = prediction_loss
                if self.method == "least_squares":
                    batch_loss = batch_loss + self.regularization_parameter * norm_loss
                if self.average_function is not None:
                    batch_loss = batch_loss + average_function_loss
                
                # accumulate epoch loss
                epoch_loss += batch_loss.item()
                
                # backprop with gradient clipping
                batch_loss.backward()
                if (batch_idx + 1) % self.gradient_accumulation == 0 or batch_idx == n_batches - 1:
                    norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                    self.opt.step()
                    self.opt.zero_grad()
                
                # Explicit memory cleanup
                del example_xs, example_ys, query_xs, query_ys, representation, y_hats
                if expected_yhats is not None:
                    del expected_yhats
                if gram is not None:
                    del gram
                
                # Force garbage collection every few batches to free memory
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # callbacks for batch
                if callback is not None:
                    # Create local variables for callback
                    callback_info ={'self': self}
                    callback.on_step(callback_info)
                    del batch_loss
            
            # Restore original n_functions
            dataset.n_functions = original_n_functions
            
            # Store average epoch loss
            avg_epoch_loss = epoch_loss / n_batches
            losses.append(avg_epoch_loss)
            
            # Update progress bar with loss info
            if progress_bar and hasattr(bar, 'set_postfix'):
                bar.set_postfix({'loss': f'{avg_epoch_loss:.6f}', 'batches': n_batches})
            
        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())
        
        return losses

    def _param_string(self):
        """ Returns a dictionary of hyperparameters for logging."""
        params = {}
        params["input_size"] = self.input_size
        params["output_size"] = self.output_size
        params["n_basis"] = self.n_basis
        params["method"] = self.method
        params["model_type"] = self.model_type
        params["regularization_parameter"] = self.regularization_parameter
        for k, v in self.model_kwargs.items():
            params[k] = v
        params = {k: str(v) for k, v in params.items()}
        return params

    @staticmethod
    def predict_number_params(input_size:Tuple[int],
                             output_size:Tuple[int],
                             n_basis:int=100,
                             model_type:Union[str, type]="MLP",
                             model_kwargs:Dict[str, Any]=dict(),
                             use_residuals_method: bool = False,
                             *args: Any, **kwargs: Any):
        """ Predicts the number of parameters in the function encoder.
        Useful for ensuring all experiments use the same number of params"""
        n_params = 0
        if model_type == "MLP":
            n_params += MLP.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += MLP.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=False, **model_kwargs)
        elif model_type == "ParallelMLP":
            n_params += ParallelMLP.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += ParallelMLP.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        elif model_type == "Euclidean":
            n_params += Euclidean.predict_number_params(output_size, n_basis)
            if use_residuals_method:
                n_params += Euclidean.predict_number_params(output_size, n_basis)
        elif model_type == "CNN":
            n_params += CNN.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += CNN.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        elif model_type == "NeuralODE":
            n_params += NeuralODE.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += NeuralODE.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        elif model_type == "ITransformer":
            n_params += ITransformer.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += ITransformer.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        elif model_type == "BA_NODE":
            n_params += BA_NODE.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += BA_NODE.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        elif isinstance(model_type, type):
            n_params += model_type.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=True, **model_kwargs)
            if use_residuals_method:
                n_params += model_type.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        else:
            raise ValueError(f"Unknown model type: '{model_type}'. Should be one of 'MLP', 'ParallelMLP', 'Euclidean', 'CNN', 'NeuralODE', 'ITransformer', or 'BA_NODE'")

        return n_params

    def forward_basis_functions(self, xs:torch.tensor) -> torch.tensor:
        """ Forward pass of the basis functions. """
        return self.model.forward(xs)

    def forward_average_function(self, xs:torch.tensor) -> torch.tensor:
        """ Forward pass of the average function. """
        return self.average_function.forward(xs) if self.average_function is not None else None
    
    def save(self, path:str):
        """ Save the function encoder to a file. """
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """ Load the function encoder from a file. """
        state_dict = torch.load(path, map_location=self.device)
        try:
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load checkpoint due to size mismatch. "
                "Ensure the saved config (history_length, n_basis, encoder settings) matches the checkpoint."
            ) from exc
        
        safe_missing_prefixes = (
            'average_function.',           # residual branch was not trained
            'model.project_basis_functions', # older checkpoints without projection head
            'project_basis_functions'
        )
        
        tolerated_missing = [
            k for k in missing_keys
            if any(k.startswith(prefix) for prefix in safe_missing_prefixes)
        ]
        critical_missing = [
            k for k in missing_keys
            if k not in tolerated_missing
        ]
        
        if tolerated_missing:
            print(f"WARNING: Missing {len(tolerated_missing)} keys that will be randomly initialized: {tolerated_missing[:5]}...")
        
        if critical_missing:
            raise RuntimeError(f"Missing critical model weights: {critical_missing[:5]}...")
        
        if unexpected_keys:
            print(f"INFO: Found {len(unexpected_keys)} unexpected keys in saved model.")
            print("This is usually fine - saved model may have extra components.")
