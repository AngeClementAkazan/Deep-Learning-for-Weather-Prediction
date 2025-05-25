## Define the TimeSeriesDataset class
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
# import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import seaborn as sns
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#The part has been inspired by  Efficient KAN: https://github.com/Blealtan/efficient-kan and TKAN: https://github.com/remigenet/TKAN

#### General Functions


class EarlyStopping:
    def __init__(self, patience=50, mode='min', verbose=False, delta=0, save_path=None):
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.delta = delta  # Added delta to ignore small improvements
        self.save_path = save_path  # Path to save the best model
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.early_stop = False
        self.counter = 0

    def __call__(self, score, model=None):
        if self.mode == 'min':
            # Check if the score has decreased significantly
            if score < self.best_score - self.delta:
                self.best_score = score
                self.counter = 0
                if self.verbose:
                    print(f'Validation loss decreased ({self.best_score:.6f} --> {score:.6f}). Saving model...')
                if model and self.save_path:
                    torch.save(model.state_dict(), self.save_path)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f'Early stopping triggered after {self.patience} epochs with no improvement.')

        elif self.mode == 'max':
            # Check if the score has increased significantly
            if score > self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
                if self.verbose:
                    print(f'Validation metric increased ({self.best_score:.6f} --> {score:.6f}). Saving model...')
                if model and self.save_path:
                    torch.save(model.state_dict(), self.save_path)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f'Early stopping triggered after {self.patience} epochs with no improvement.')

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, n_steps_in, n_steps_out,target_names):
        """Initialize the dataset with a pandas DataFrame.
        Parameters:
        - dataframe: Pandas DataFrame containing the time series data.
        - n_steps_in: Number of time steps for input.
        - n_steps_out: Number of time steps for output.
        -target_names: NAmes of the target variables.
        """
        self.dataframe = dataframe
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out   
        self.target_names=target_names   
        # Prepare the input and output sequences
        self.X, self.y = self.sliding_window(self.dataframe, self.n_steps_in, self.n_steps_out,self.target_names)
    def sliding_window(self, dataframe, n_steps_in, n_steps_out,target_names):
        X, y = list(), list()  
        for i in range(len(dataframe)):
            end_idx = i + n_steps_in
            out_end_idx = end_idx + n_steps_out - 1
            # Ensure valid index range
            if out_end_idx >= len(dataframe):
                break
            # Extract X and y sequences
            seq_x = dataframe.iloc[i:end_idx, :].values  # X sequence
            seq_y = dataframe[target_names][end_idx:out_end_idx + 1].values  # y sequence (last column)| Redefine this!!!
            # Check if `seq_y` is shorter than `n_steps_out`
            if len(seq_y) < n_steps_out:
                pad_length = n_steps_out - len(seq_y)
                avg_value = np.mean(seq_y)  # Calculate the average of `seq_y`
                # Pad `seq_y` with the average value
                seq_y = np.pad(seq_y, (0, pad_length), 'constant', constant_values=avg_value)
            # Ensure `seq_y` is a 2D array
            if len(seq_y.shape) == 1:
                seq_y = seq_y.reshape(-1, 1)
            # Append the sequences to the list
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        # Return the input-output pair for the given index
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
    
 #KAN and TKAN
def kan_prediction_plots(dict_m, cities, scalers,model_name="KAN",var_to_pred = None,obs_length=14):
    metrics_list = []
    for idx_d,city in enumerate(cities):
         # Create subplots for visualization
        scaler=scalers[idx_d]
        fig, axes = plt.subplots( 3,2, figsize=(12, 10))
        fig.suptitle(f"{city}  {model_name} Predictions vs. Actuals ", fontsize=14, fontweight="bold")
        for col, var_name in enumerate(var_to_pred):
            model = dict_m[city][0][var_name]
            model.eval().to(device)
            test_loader = dict_m[city][1][var_name]
            dates = dict_m[city][2][var_name][-1]
            
            y_true_list = []
            y_pred_list = []
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device).squeeze()
                    if model_name!="KAN":
                        output,_ = model(X_batch)
                        outputs=output[:, -1].reshape(-1, 1)
                    else:
                        outputs= model(X_batch)

                    y_true_list.append(y_batch.cpu().numpy())
                    y_pred_list.append(outputs.cpu().numpy())
            # print("Xbatch shape:", X_batch.shape)

            # Convert lists to NumPy arrays
            y_true = f(np.concatenate(y_true_list))
            y_pred = f(np.concatenate(y_pred_list))
            # print("y_true:",y_true[:10], "y_pred:",y_pred[:10])
            # Inverse transform
            dummy_features = np.zeros((y_true.shape[0], X_batch.shape[1] ))
            actuals_padded = np.hstack([dummy_features, y_true.reshape(-1, 1)])
            predictions_padded = np.hstack([dummy_features, y_pred.reshape(-1, 1)])
            # print("actuals_padded shape:",actuals_padded.shape, "Xbatch shape:", X_batch.shape,"predictions_padded shape:",predictions_padded.shape)
            actuals_ = scaler[col].inverse_transform(actuals_padded)[:, -1]
            predictions_ = scaler[col].inverse_transform(predictions_padded)[:, -1]
            # print( predictions_ [:10],actuals_ [:10]) 
            # Compute metrics
            mse_ = np.mean((actuals_ - predictions_) ** 2)
            rmse_ = np.sqrt(mse_)
            mae_ = np.mean(np.abs(actuals_ - predictions_))
            r2_ = 1 - (np.sum((actuals_ - predictions_) ** 2) / np.sum((actuals_ - np.mean(actuals_)) ** 2))

            mask = actuals_ != 0
            mape_ = np.mean(np.abs((actuals_[mask] - predictions_[mask]) / actuals_[mask])* 100) if mask.any() else np.nan

            # Store metrics
            metrics_list.append({
                "City": city,
                "Model": f"{model_name}",
                "Variable": var_name,
                "MSE": round(mse_,4),
                "RMSE": round(rmse_,4),
                "MAE": round(mae_,4),
                "R²": round(r2_,4),
                "MAPE": round(mape_,4)
            })
            # --- Subplot 1: Line Plot ---
            # dates=
            axes[col,0].plot(dates, actuals_, label="Actual", color="blue", alpha=0.7)
            axes[col,0].plot(dates, predictions_, label="Predicted", color="red", linestyle="dashed", alpha=0.7)
            axes[col,0].set_title(f"{var_name} - Time Series")
            axes[col,0].set_xlabel("Time Step")
            axes[col,0].set_ylabel(var_name)
            axes[col,0].legend(framealpha=0.5)
            axes[col,0].xaxis.set_major_locator(mdates.DayLocator(interval=200))  # Show every 7th day
            axes[col,0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format: YYYY-MM-DD
            # axes[col, 0].tick_params(axis="x", rotation=45)  # ✅ Correct


            # --- Subplot 2: Histogram ---
            sns.histplot(actuals_, label="Actual", color="blue", kde=True, bins=30, alpha=0.5, ax=axes[col,1])
            sns.histplot(predictions_, label="Predicted", color="red", kde=True, bins=30, alpha=0.5, ax=axes[col,1])
            axes[col,1].set_title(f"{var_name} - Distribution")
            axes[col,1].set_xlabel(var_name)
            axes[col,1].set_ylabel("Frequency")
            axes[col,1].legend(framealpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
        plt.savefig(f"{city}_{var_name}_{model_name}.png")
        plt.show()
        # print(idx_d)
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df
######   Deep RNNs Functions

def Make_tidydata_deepRNN(data, scaler_other, scaler_prec):
    data = data.sort_index()
    # Extract the precipitation column
    prec = data.pop("PREC") 
    # other_ys=data.pop(("T2M","PS")) 
    extracted_cols = data[["T2M", "PS"]].copy()  # make a copy if you want to keep it
    data.drop(columns=["T2M", "PS"], inplace=True)  # remove from original 
    data_concat = pd.concat([data, extracted_cols], axis=1)
    # Fit both scalers on the entire dataset BEFORE splitting
    # print(data_concat.shape, prec.shape)
    prec_scaled = scaler_prec.fit_transform(prec.values.reshape(-1, 1))
    data_scaled = scaler_other.fit_transform(data_concat)
    # Convert scaled data back to DataFrame
    data_scaled = pd.DataFrame(data_scaled, columns=data_concat.columns, index=data.index)
    prec_scaled = pd.DataFrame(prec_scaled, columns=["PREC"], index=data.index)
    # Concatenate scaled data
    data_final = pd.concat([data_scaled, prec_scaled], axis=1)

    # Splitting the dataset into training, validation, and test sets
    train_size = int(len(data_final) * 0.8)
    val_size = int(train_size * 0.10)

    Xy_train = data_final.iloc[:train_size - val_size].copy()
    Xy_val = data_final.iloc[train_size - val_size:train_size].copy()
    Xy_test = data_final.iloc[train_size:].copy()

    return Xy_train, Xy_val, Xy_test
  
# Create data loaders
def create_data_loaders(TimeSeriesDataset,Make_tidydata,df,target_names,scaler_other, scaler_prec,window_length,obs_length,batch_size):
    Xy_train,Xy_val,Xy_test=Make_tidydata(df,scaler_other, scaler_prec)
    # Creating time series data
    train_dataset = TimeSeriesDataset(Xy_train,obs_length, window_length,target_names)
    val_dataset = TimeSeriesDataset(Xy_val, obs_length, window_length,target_names)
    test_dataset = TimeSeriesDataset(Xy_test, obs_length, window_length,target_names)
    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,val_loader,test_loader

####### KAN functions

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1) # Create n grid for n features
            .contiguous() #ensures that the tensor is stored in memory in a contiguous manner, which improves performance for operations that require sequential memory access.
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
# We don't know out_features (Guess:2*in_features +1), the reason behind the per-batch structuration of self.spline_weight and the structuration of self.base_weight
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()
# The parameters abbove are not really known
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                ) # Provide initial best values to the self.spline_weight (either scaled or not )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).
        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        # print("A shape:", A.shape)  # Should be (batch, n, m)
        # print("B shape:", B.shape)  # Should be (batch, n, out_features)

        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )
    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
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

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    
######## TKAN functions


class DropoutRNNCell:
    def __init__(self, dropout: float = 0.0, recurrent_dropout: float = 0.0):
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def get_dropout_mask(self, step_input):
        if self._dropout_mask is None and self.dropout > 0:
            self._dropout_mask = F.dropout(torch.ones_like(step_input), p=self.dropout, training=True)
        return self._dropout_mask

    def get_recurrent_dropout_mask(self, step_input):
        if self._recurrent_dropout_mask is None and self.recurrent_dropout > 0:
            self._recurrent_dropout_mask = F.dropout(torch.ones_like(step_input), p=self.recurrent_dropout, training=True)
        return self._recurrent_dropout_mask

    def reset_dropout_mask(self):
        self._dropout_mask = None

    def reset_recurrent_dropout_mask(self):
        self._recurrent_dropout_mask = None

class TKANCell(nn.Module, DropoutRNNCell):
    def __init__(self, units,layers_hidden,max_order=6,  tkan_activations=None, activation="tanh", recurrent_activation="sigmoid", 
                 use_bias=True, dropout=0.0, recurrent_dropout=0.0,training=False):
        super(TKANCell, self).__init__()
        DropoutRNNCell.__init__(self, dropout, recurrent_dropout)
        # self.input_dim=layers_hidden[0]
        self.units = units
        self.activation = getattr(torch, activation)
        self.recurrent_activation = getattr(torch, recurrent_activation)
        self.use_bias = use_bias
        # self.tkan_sub_layers = tkan_sub_layers
        self.sub_kan_input_dim =layers_hidden[0]
        self.sub_kan_output_dim =layers_hidden[1]
        
        # Note: recurrent_regularizer and recurrent_constraint are not directly supported in PyTorch
        # sub_tkan_recurrent_kernel_input
        self.kernel = nn.Parameter(torch.Tensor(self.sub_kan_input_dim , 3 * units))
        self.recurrent_kernel = nn.Parameter(torch.Tensor(units, 3 * units))
        self.bias = nn.Parameter(torch.Tensor(3 * units)) if use_bias else None

        self.tkan_sub_layers =nn.ModuleList([tkan_activations(layers_hidden,spline_order=3)  ]) #for i in range(1,max_order) if #nn.ModuleList([tkan_activations(self.input_dim, 2*self.input_dim+1,spline_order=i) for i in range(1,6) ])#or [BSplineActivation(3)  for _ in range(3)
        #self.tkan_sub_layers = tkan_activations(self.input_dim, 2*self.input_dim+1)
        self.num_sub_layers = len(self.tkan_sub_layers)
        #self.num_sub_layers = 2*self.input_dim+1
        # 

        # Define parameters equivalent to add_weight calls in TensorFlow/Keras
        self.sub_tkan_kernel = nn.Parameter(torch.Tensor(self.num_sub_layers, self.sub_kan_output_dim * 2))
        self.sub_tkan_recurrent_kernel_inputs = nn.Parameter(
            torch.Tensor(self.num_sub_layers,self.sub_kan_input_dim  , self.sub_kan_input_dim)
        )
        self.sub_tkan_recurrent_kernel_states = nn.Parameter(
            torch.Tensor(self.num_sub_layers, self.sub_kan_output_dim, self.sub_kan_input_dim) # It is three dimensional because we concatenate the L (num_sub_layers) sublayer outputs with shape (Kan_out,Kan_in)
        )
        self.aggregated_weight = nn.Parameter(
            torch.Tensor(self.num_sub_layers * self.sub_kan_output_dim, units) # (L*Kan_out) units output dimension
        )
        self.aggregated_bias = nn.Parameter(torch.Tensor(units))
        
        self._init_parameters()
        

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.xavier_uniform_(self.sub_tkan_recurrent_kernel_states )
        nn.init.xavier_uniform_(self.sub_tkan_kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.orthogonal_(self.sub_tkan_recurrent_kernel_inputs)  
        nn.init.xavier_uniform_(self.aggregated_weight)

        if self.use_bias:
            nn.init.zeros_(self.bias)
            nn.init.zeros_(self.aggregated_bias)

    def forward(self, inputs, states):
        # print(inputs.size())
        batch_size,_ = inputs.size()
        h_tm1, c_tm1, *sub_states = states  # The * operator here means "pack any remaining elements into a list"

        dp_mask = self.get_dropout_mask(inputs)
        rec_dp_mask = self.get_recurrent_dropout_mask(h_tm1)

        if self.training and self.dropout > 0.0:
            inputs = inputs * dp_mask
        if self.training and self.recurrent_dropout > 0.0:
            h_tm1 = h_tm1 * rec_dp_mask
        # Process each sub-layer
        # Preallocate tensors for sub-layer outputs and new states
        if self.use_bias:
            x = self.recurrent_activation(torch.matmul(inputs, self.kernel) +
                                  torch.matmul(h_tm1, self.recurrent_kernel) +
                                  self.bias)
        else:
            x = self.recurrent_activation(torch.matmul(inputs, self.kernel) +
                                          torch.matmul(h_tm1, self.recurrent_kernel))
        x_i, x_f, x_c = torch.chunk(x, 3, dim=-1)
        # Initialize lists to store outputs and new states
        sub_outputs = []
        new_sub_states = []
        # cpt
        # Loop over the sub-layers and corresponding sub-states
        # print("len:",len(self.tkan_sub_layers))
        #sub_layer_output = self.tkan_sub_layers(inputs)
        #count1 = 0
        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            # print(sub_layer[:2])
            # Get the kernels for input and state from the corresponding lists
            sub_kernel_x = self.sub_tkan_recurrent_kernel_inputs[idx]
            sub_kernel_h = self.sub_tkan_recurrent_kernel_states[idx]
            # print("shapes:",inputs.size(),sub_kernel_x.size(),sub_state.size(),sub_kernel_h.size(),( sub_state @ sub_kernel_h.T).size(),(inputs @ sub_kernel_x).size())
            # Compute the aggregated input: inputs @ sub_kernel_x + sub_state @ sub_kernel_h
            # shapes=[a.shape for a in sub_states]
            # print("inputs.shape, sub_kernel_x.shape,sub_state.shape, sub_kernel_h.shape:",inputs.shape, sub_kernel_x.shape,sub_state.shape, sub_kernel_h.shape,self.sub_kan_output_dim)
            agg_input = inputs @ sub_kernel_x + sub_state @ sub_kernel_h
            # if torch.isnan(inputs).any():
            #     print("NaN in inputs!")
            # print("sub_kernel_x:",sub_kernel_x[:2],"sub_kernel_h:",sub_kernel_h[:2])
            # dim=(64,10)*(10,skid)+(64,skod)*(skod,skid)=(64,skid)
            # Pass through the sub-layer
            sub_output = sub_layer(agg_input)#dim=(64,skod)
            # print(sub_output.shape,sub_output[:4])
            # Split the sub-kernel into two parts along dimension 0
            # (Assuming the first dimension is even so that it can be split equally)
            sub_recurrent_kernel_h, sub_recurrent_kernel_x = torch.chunk(self.sub_tkan_kernel[idx], 2, dim=-1) # each has dimension (skod)
            #print("shapes:",sub_recurrent_kernel_h.size(),sub_output.size(),sub_state.size(),sub_recurrent_kernel_x.size())
            # Compute the new sub-state
            # " Next Task : Check the dimension of the sub_state "
            new_sub_state = sub_recurrent_kernel_h*sub_output  + sub_state * sub_recurrent_kernel_x
                                #dim=(64,skod)*(skod)  +  (64,skod)*((skod)
    
            # Append the computed sub_output and new_sub_state to the lists
            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)
            # count1 =+1
            # print("count :", count1)
            # print(len(new_states)
        # Stack the sub-layer outputs into a single tensor.
        # If each sub_output has shape (batch_size, sub_dim), stacking along dim=1 yields:
        # shape (batch_size, num_sub_layers, sub_dim)
        sub_outputs = torch.stack(sub_outputs, dim=1) # (bs, skod*num_sub_layers)
        # print("sub_outputs:",sub_outputs.shape)
        # Reshape to aggregate sub-layer outputs: (batch_size, -1)
        batch_size = inputs.shape[0]
        aggregated_sub_output = sub_outputs.reshape(batch_size, -1)
        
        # Aggregate using weights and bias
        aggregated_input = torch.matmul(aggregated_sub_output, self.aggregated_weight) + self.aggregated_bias
                            #(bs, skod*num_sub_layers) @ (skod*num_sub_layers,units)+    ( units) .reshape(sub_kan_output_dim,-1)
        #aggregated_input has (bs, units=1)
        xo = self.recurrent_activation(aggregated_input) #(bs, units=1)

        c = x_f * c_tm1 + x_i * x_c

        # # Compute the TKAN cell's new states
        h = xo * self.activation(c) #(?bs,units=1)

        # Prepare output and new states
        new_states = [h, c] + list(torch.unbind(torch.stack(new_sub_states)))
        return h, new_states    

    def get_initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.units, device=device)
        c = torch.zeros(batch_size, self.units, device=device)
        sub_states = [torch.zeros(batch_size, self.sub_kan_output_dim, device=device) for _ in range(self.num_sub_layers)]
        return [h, c] + list(sub_states)

class TKAN(nn.Module):
    def __init__(self, units, layers_hidden,max_order=6, tkan_activations=None, activation="tanh", recurrent_activation="sigmoid", use_bias=True, 
                 dropout=0.0, recurrent_dropout=0.0):
        super(TKAN, self).__init__()
        self.cell = TKANCell(units,layers_hidden,max_order, tkan_activations, activation, recurrent_activation, use_bias, dropout, recurrent_dropout)

    def forward(self, inputs, initial_state=None):
        # print("input size:",inputs.size())
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device
        if initial_state is None:
            initial_state = self.cell.get_initial_state(batch_size, device)
        
        states = initial_state
        outputs = []
        for t in range(seq_len):
            output, states = self.cell(inputs[:, t, :], states)
            outputs.append(output.unsqueeze(1))
        
        return torch.cat(outputs, dim=1), states #final ouput (bs, seq_len)
