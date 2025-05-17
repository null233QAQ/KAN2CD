import torch
import torch.nn.functional as F
import math


class KLayer(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            enable_standalone_scale_spline: bool = True,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: list = [-1, 1],
    ):
        super(KLayer, self).__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.grid_segments = grid_size
        self.spline_degree = spline_order
        self.noise_scale_factor = scale_noise
        self.base_weight_scale = scale_base
        self.spline_weight_scale = scale_spline
        self.use_separate_spline_scaler = enable_standalone_scale_spline
        self.base_fn = base_activation()
        self.grid_point_epsilon = grid_eps

        # Initialize grid
        grid_step = (grid_range[1] - grid_range[0]) / self.grid_segments
        # Construct grid points, extending beyond the main range for spline calculations
        knot_vector = (
                torch.arange(-self.spline_degree, self.grid_segments + self.spline_degree + 1) * grid_step
                + grid_range[0]
        )
        # Expand grid for each input feature
        expanded_grid = knot_vector.expand(self.input_dim, -1).contiguous()
        self.register_buffer("grid_knots", expanded_grid)

        # Initialize learnable parameters
        self.base_weight_param = torch.nn.Parameter(torch.empty(self.output_dim, self.input_dim))
        self.spline_weight_param = torch.nn.Parameter(
            torch.empty(self.output_dim, self.input_dim, self.grid_segments + self.spline_degree)
        )
        if self.use_separate_spline_scaler:
            self.spline_scaler_param = torch.nn.Parameter(
                torch.empty(self.output_dim, self.input_dim)
            )

        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, in_features)
        if not (x.dim() == 2 and x.size(1) == self.input_dim):
            raise ValueError(
                f"Input tensor must have shape (batch_size, {self.input_dim}), "
                f"but got {x.shape}"
            )

        # Base component computation
        activated_base = self.base_fn(x)
        base_component = F.linear(activated_base, self.base_weight_param)

        # Spline component computation
        # b_spline_values shape: (batch_size, in_features, grid_size + spline_order)
        b_spline_values = self._calculate_b_spline_basis(x)
        # Flatten for linear layer: (batch_size, in_features * (grid_size + spline_order))
        flattened_spline_basis = b_spline_values.view(x.size(0), -1)

        # effective_spline_coeffs shape: (out_features, in_features, grid_size + spline_order)
        # CORRECTED: Access property without parentheses
        effective_spline_coeffs = self._get_effective_spline_weights
        # Flatten for linear layer: (out_features, in_features * (grid_size + spline_order))
        flattened_spline_coeffs = effective_spline_coeffs.view(self.output_dim, -1)

        spline_component = F.linear(flattened_spline_basis, flattened_spline_coeffs)

        return base_component + spline_component

    def _calculate_b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Computes B-spline basis functions.
        x: input tensor, shape (batch_size, in_features)
        Returns: B-spline bases, shape (batch_size, in_features, grid_size + spline_order)
        """
        if not (x.dim() == 2 and x.size(1) == self.input_dim):
            raise ValueError(
                f"Input for B-spline basis calculation must have shape (batch_size, {self.input_dim}), "
                f"but got {x.shape}"
            )

        # grid_knots: (in_features, grid_size + 2 * spline_order + 1)
        # x_reshaped: (batch_size, in_features, 1) for broadcasting
        x_reshaped = x.unsqueeze(-1)

        # Initialize basis functions of order 0
        # basis_functions shape: (batch_size, in_features, grid_size + 2 * spline_order)
        basis_functions = (
                (x_reshaped >= self.grid_knots[:, :-1]) & (x_reshaped < self.grid_knots[:, 1:])
        ).to(x.dtype)

        # Cox-de Boor recursion formula
        for k_order in range(1, self.spline_degree + 1):
            # First term of the recursion
            numerator1 = x_reshaped - self.grid_knots[:, : -(k_order + 1)]
            denominator1 = self.grid_knots[:, k_order:-1] - self.grid_knots[:, : -(k_order + 1)]
            # Add small epsilon to prevent division by zero if knots are identical
            term1 = (numerator1 / (denominator1 + 1e-8)) * basis_functions[:, :, :-1]

            # Second term of the recursion
            numerator2 = self.grid_knots[:, k_order + 1:] - x_reshaped
            denominator2 = self.grid_knots[:, k_order + 1:] - self.grid_knots[:, 1:(-k_order)]
            # Add small epsilon
            term2 = (numerator2 / (denominator2 + 1e-8)) * basis_functions[:, :, 1:]

            basis_functions = term1 + term2

        # Final shape: (batch_size, in_features, grid_size + spline_order)
        if not basis_functions.size() == (
                x.size(0),
                self.input_dim,
                self.grid_segments + self.spline_degree,
        ):
            raise RuntimeError(
                "Mismatch in B-spline basis functions' final shape."
            )
        return basis_functions.contiguous()

    @property
    def _get_effective_spline_weights(self) -> torch.Tensor:
        """Returns spline_weight_param, potentially scaled by spline_scaler_param."""
        if self.use_separate_spline_scaler:
            # Unsqueeze to allow broadcasting: (out_features, in_features, 1)
            return self.spline_weight_param * self.spline_scaler_param.unsqueeze(-1)
        return self.spline_weight_param

    @torch.no_grad()
    def update_grid_from_data(self, x: torch.Tensor, margin: float = 0.01):
        """Updates grid_knots based on the distribution of input data `x`."""
        if not (x.dim() == 2 and x.size(1) == self.input_dim):
            raise ValueError(f"Input for grid update must have shape (batch_size, {self.input_dim})")

        current_batch_size = x.size(0)

        # Calculate spline activations for the current input
        # b_spline_activations: (batch, in_features, coeff_dim)
        b_spline_activations = self._calculate_b_spline_basis(x)
        # Transpose for matrix multiplication: (in_features, batch, coeff_dim)
        b_spline_activations_t = b_spline_activations.permute(1, 0, 2)

        # Get current spline coefficients
        # current_coeffs: (out_features, in_features, coeff_dim)
        current_coeffs = self._get_effective_spline_weights  # Access property
        # Transpose for matrix multiplication: (in_features, coeff_dim, out_features)
        current_coeffs_t = current_coeffs.permute(1, 2, 0)

        # Compute unreduced spline outputs (outputs before summing over out_features)
        # spline_outputs_unreduced_t: (in_features, batch, out_features)
        spline_outputs_unreduced_t = torch.bmm(b_spline_activations_t, current_coeffs_t)
        # Transpose back: (batch, in_features, out_features)
        spline_outputs_unreduced = spline_outputs_unreduced_t.permute(1, 0, 2)

        # Sort input data for adaptive grid generation
        sorted_x_data = torch.sort(x, dim=0)[0]

        # Define indices for adaptive grid points
        adaptive_indices = torch.linspace(
            0, current_batch_size - 1, self.grid_segments + 1, dtype=torch.int64, device=x.device
        )
        adaptive_grid_points = sorted_x_data[adaptive_indices]  # (grid_size + 1, in_features)

        # Define uniform grid component
        data_range = sorted_x_data[-1] - sorted_x_data[0]
        uniform_grid_step = (data_range + 2 * margin) / self.grid_segments

        # uniform_grid_points: (grid_size + 1, 1) for arange part, then broadcasted by uniform_grid_step
        # Resulting shape: (grid_size + 1, in_features)
        uniform_grid_points = (
                torch.arange(self.grid_segments + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
                * uniform_grid_step + sorted_x_data[0] - margin
        )

        # Combine adaptive and uniform grids
        # combined_grid_core: (grid_size + 1, in_features)
        combined_grid_core = (
                self.grid_point_epsilon * uniform_grid_points
                + (1 - self.grid_point_epsilon) * adaptive_grid_points
        )

        # Extend grid for spline order (knot vector requires points outside the data range)
        # lower_extensions: (spline_order, in_features)
        lower_extensions = combined_grid_core[0:1] - uniform_grid_step * torch.arange(
            self.spline_degree, 0, -1, device=x.device
        ).unsqueeze(1)

        # upper_extensions: (spline_order, in_features)
        upper_extensions = combined_grid_core[-1:] + uniform_grid_step * torch.arange(
            1, self.spline_degree + 1, device=x.device
        ).unsqueeze(1)

        # Concatenate to form the new complete grid
        # new_knot_vector: (grid_size + 2 * spline_order + 1, in_features)
        new_knot_vector = torch.cat([lower_extensions, combined_grid_core, upper_extensions], dim=0)

        self.grid_knots.copy_(new_knot_vector.T)  # Transpose to (in_features, num_knots)

        # Update spline weights based on the new grid and previous outputs
        # This aims to preserve the function shape as much as possible
        new_coeffs = self._calculate_coeffs_from_curve(x, spline_outputs_unreduced)
        self.spline_weight_param.data.copy_(new_coeffs)

    def _calculate_coeffs_from_curve(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes spline coefficients that interpolate given (x, y) points using the current grid.
        x: input, shape (batch_size, in_features)
        y: target output, shape (batch_size, in_features, out_features)
        Returns: coefficients, shape (out_features, in_features, grid_size + spline_order)
        """
        if not (x.dim() == 2 and x.size(1) == self.input_dim):
            raise ValueError(f"Input x for curve2coeff must have shape (batch_size, {self.input_dim})")
        if not (y.size() == (x.size(0), self.input_dim, self.output_dim)):
            raise ValueError(
                f"Input y for curve2coeff must have shape (batch_size, {self.input_dim}, {self.output_dim})")

        # Basis matrix A: (in_features, batch_size, grid_size + spline_order)
        # Calculated using the new grid implicitly via _calculate_b_spline_basis
        basis_matrix_A = self._calculate_b_spline_basis(x).transpose(0, 1)

        # Target values B: (in_features, batch_size, out_features)
        target_matrix_B = y.transpose(0, 1)

        # Solve AX = B for X (coefficients) using least squares for each feature independently
        # lstsq solution shape: (in_features, grid_size + spline_order, out_features)
        try:
            # torch.linalg.lstsq returns a named tuple, solution is the first element
            solved_coefficients = torch.linalg.lstsq(basis_matrix_A, target_matrix_B).solution
        except torch.linalg.LinAlgError as e:
            # Fallback or error handling if lstsq fails (e.g., singular matrix)
            # For simplicity, we'll re-raise, but a production system might try a pseudo-inverse or Tikhonov regularization
            raise RuntimeError(f"torch.linalg.lstsq failed: {e}")

        # Permute to match spline_weight_param shape: (out_features, in_features, grid_size + spline_order)
        final_coefficients = solved_coefficients.permute(2, 0, 1)

        if not final_coefficients.size() == (
                self.output_dim,
                self.input_dim,
                self.grid_segments + self.spline_degree,
        ):
            raise RuntimeError("Mismatch in calculated coefficients' shape.")

        return final_coefficients.contiguous()

    def initialize_weights(self):
        """Initializes learnable parameters."""
        # Initialize base weights using Kaiming uniform for SiLU-like activations
        torch.nn.init.kaiming_uniform_(self.base_weight_param, a=math.sqrt(5) * self.base_weight_scale)

        with torch.no_grad():
            # Initialize spline_weight_param with small random noise
            # This noise represents the initial "function values" at grid points for fitting
            noise_dimensions = (self.grid_segments + 1, self.input_dim, self.output_dim)
            # Uniform noise in [-0.5, 0.5] scaled by noise_scale_factor / grid_segments
            random_noise_values = (
                    (torch.rand(noise_dimensions, device=self.base_weight_param.device) - 0.5)
                    * self.noise_scale_factor / self.grid_segments
            )

            # Grid points for initial curve fitting (core grid, excluding extensions for spline order)
            # init_grid_subset shape: (grid_size + 1, in_features)
            # self.grid_knots is (in_features, num_total_knots)
            # We need to select the 'core' grid points corresponding to grid_segments + 1 points
            # These are from index spline_degree to spline_degree + grid_segments
            core_grid_indices = slice(self.spline_degree, self.spline_degree + self.grid_segments + 1)
            init_grid_subset = self.grid_knots[:, core_grid_indices].T.contiguous()

            coeff_init_scale = self.spline_weight_scale
            if self.use_separate_spline_scaler:
                # If using a separate scaler, initialize base spline weights with scale 1
                coeff_init_scale = 1.0

                # Calculate initial spline coefficients by fitting to the noisy values
            initial_spline_coeffs = self._calculate_coeffs_from_curve(
                init_grid_subset, random_noise_values
            )
            self.spline_weight_param.data.copy_(coeff_init_scale * initial_spline_coeffs)

            if self.use_separate_spline_scaler:
                # Initialize spline scaler using Kaiming uniform
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler_param, a=math.sqrt(5) * self.spline_weight_scale
                )

    def regularization_loss(self, regularize_activation_strength: float = 1.0,
                            regularize_entropy_strength: float = 1.0) -> torch.Tensor:
        """
        Computes a regularization loss.
        This loss encourages sparsity (L1-like) and diversity (entropy-like) of spline activations.
        """
        epsilon = 1e-7  # Small constant for numerical stability

        # Mean absolute spline weights per output-input feature pair
        # mean_abs_coeffs shape: (out_features, in_features)
        mean_abs_coeffs = self.spline_weight_param.abs().mean(dim=-1)

        # Activation regularization (L1-like on mean absolute coefficients)
        activation_reg_term = mean_abs_coeffs.sum()

        # Entropy regularization on the distribution of mean absolute coefficients
        # Normalized probabilities p: (out_features, in_features)
        norm_mean_abs_coeffs = mean_abs_coeffs / (activation_reg_term + epsilon)
        # Entropy term: -sum(p * log(p))
        entropy_reg_term = -torch.sum(norm_mean_abs_coeffs * torch.log(norm_mean_abs_coeffs + epsilon))

        return (
                regularize_activation_strength * activation_reg_term
                + regularize_entropy_strength * entropy_reg_term
        )


class QKAN(torch.nn.Module):
    def __init__(
            self,
            layer_dims: list,  # e.g., [input_dim, hidden1_dim, hidden2_dim, output_dim]
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: list = [-1, 1],
    ):
        super(QKAN, self).__init__()

        self.network_layers = torch.nn.ModuleList()
        # Create EKAN layers sequentially based on layer_dims
        for idx in range(len(layer_dims) - 1):
            current_in_features = layer_dims[idx]
            current_out_features = layer_dims[idx + 1]
            self.network_layers.append(
                KLayer(
                    in_features=current_in_features,
                    out_features=current_out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grids_dynamically: bool = False) -> torch.Tensor:
        processed_x = x
        for active_layer in self.network_layers:
            if update_grids_dynamically:
                # Update grid for the current layer based on its specific input
                active_layer.update_grid_from_data(processed_x)
            processed_x = active_layer(processed_x)
        return processed_x

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> torch.Tensor:
        accumulated_reg_loss = torch.tensor(0.0, device=self.network_layers[0].base_weight_param.device if len(
            self.network_layers) > 0 else torch.device('cpu'))
        for active_layer in self.network_layers:
            accumulated_reg_loss += active_layer.regularization_loss(
                regularize_activation, regularize_entropy
            )
        return accumulated_reg_loss