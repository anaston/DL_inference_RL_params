# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import os


class Loss:
    """
    Class for custom loss function
    """
    def __init__(self, name, weight, tol=0):
        """Initialize loss function"""
        # Name of loss function
        self.name = name

        # Loss function
        self.func = getattr(nn, f'{self.name}Loss')()

        # Relative weight of loss when calculating multi-loss
        self.weight = weight

        # Tolerance for numerical stability
        self.tol = tol

    def __call__(self, y_pred, y_true):
        """Compute loss"""
        if self.name == 'CrossEntropy':
            # Reshape y_pred to match y_true
            y_pred = torch.transpose(y_pred, 1, 2)
        elif self.name == 'BCE':
            # Add tolerance to predictions
            y_pred += self.tol

        return self.func(y_pred, y_true) * self.weight


class GRU(nn.Module):
    """
    Class for custom RNN model with two GRU layers

    NB: layer names are defined in a backwards-compatible way, so that old
    models can be also re-used
    """
    def __init__(self, ds, hidden_size, dropout, backwards_compatible=False):
        """Initialize model"""
        # Initialize parent class
        super().__init__()

        # Fields for backwards compatibility
        self.backwards_compatible = backwards_compatible

        # Initialize dictionary for model output
        self.y = dict()

        # Initialize each loss of multi-loss function
        self.loss = {'action': Loss('BCE', weight=1),
                     'alpha_bin': Loss('CrossEntropy', weight=0.2),
                     'beta_bin': Loss('CrossEntropy', weight=0.2),
                     'alpha': Loss('MSE', weight=6),
                     'beta': Loss('MSE', weight=0.1)}
        self.loss = {k: v for k, v in self.loss.items() if k in ds.outputs}

        # GRU layer on inputs (self.gru_input instead of self.hidden_0?)
        setattr(self, self._c('gru_input'),
                nn.GRU(input_size=ds.ninputs, hidden_size=hidden_size,
                       num_layers=1, batch_first=True, dropout=dropout))

        # Linear layers on GRU output
        setattr(self, self._c('linear_alpha_bin'),
                nn.Linear(hidden_size, ds.alpha_nbins))
        setattr(self, self._c('linear_beta_bin'),
                nn.Linear(hidden_size, ds.beta_nbins))

        # RELU layers on linear output
        self.relu_alpha = nn.ReLU()
        self.relu_beta = nn.ReLU()

        # Linear layers on RELU output
        setattr(self, self._c('linear_alpha'), nn.Linear(ds.alpha_nbins, 1))
        setattr(self, self._c('linear_beta'), nn.Linear(ds.beta_nbins, 1))

        # GRU layer on embedded inputs
        setattr(self, self._c('gru_embedded'),
                nn.GRU(input_size=ds.ninputs + ds.alpha_nbins + ds.beta_nbins,
                       hidden_size=hidden_size, num_layers=1, batch_first=True,
                       dropout=dropout))

        # Linear layer on embedded GRU output
        setattr(self, self._c('linear_action'),
                nn.Linear(hidden_size, ds.vars['action']['last_shape']))

    def _c(self, layer_name):
        """Return name of layer for backwards compatibility"""
        comp_dict = {'gru_input': 'hidden_0', 'gru_embedded': 'hidden_1',
                     'linear_alpha_bin': 'out_alpha',
                     'linear_beta_bin': 'out_beta',
                     'linear_alpha': 'reg_alpha', 'linear_beta': 'reg_beta',
                     'linear_action': 'out_action'}
        return comp_dict[layer_name] if self.backwards_compatible else (
            layer_name)

    def forward(self, X):
        """Forward pass model"""
        # Predict hidden estimates using GRU layer
        y_hidden, state_hidden = getattr(self, self._c('gru_input'))(X)

        # Predict binary alpha and beta estimates using linear and RELU layers
        self.y['alpha_bin'] = self.relu_alpha(getattr(
            self, self._c('linear_alpha_bin'))(y_hidden))
        self.y['beta_bin'] = self.relu_beta(getattr(
            self, self._c('linear_beta_bin'))(y_hidden))

        # Predict alpha and beta estimates using linear layer
        self.y['alpha'] = getattr(
            self, self._c('linear_alpha'))(self.y['alpha_bin'])
        self.y['beta'] = getattr(
            self, self._c('linear_beta'))(self.y['beta_bin'])

        # Create embedded input
        X_embedded = torch.cat(
            (X, self.y['alpha_bin'], self.y['beta_bin']), dim=2)

        # Predict action using GRU layer
        y_action, state_action = getattr(
            self, self._c('gru_embedded'))(X_embedded)
        self.y['action'] = F.softmax(getattr(
            self, self._c('linear_action'))(y_action), dim=-1)

    def calc_loss(self, y_true):
        """Calculate each loss of multi-loss function"""
        # Calculate losses
        return [self.loss[val](self.y[val], y_true[i])
                for i, val in enumerate(self.loss.keys())]


def train_one_epoch(model, device, optimizer, train_ds):
    """Train RNN model for one epoch"""
    # Initialize running loss as array
    running_loss = np.zeros(len(model.loss))

    # Instantiate Data Loader for traning data
    train_loader = DataLoader(train_ds, shuffle=False, batch_size=1)

    # Loop over training data
    for X, y_true in train_loader:
        # Move data to GPU
        X = X.to(device)
        y_true = [y.to(device) for y in y_true]

        # Zero the gradient buffers
        optimizer.zero_grad()

        # Forward pass
        model(X)

        # Calculate loss
        loss = model.calc_loss(y_true)

        # Backpropagate loss
        sum(loss).backward()

        # Update RNN weights
        optimizer.step()

        # Update loss
        running_loss += [val.item() for val in loss]

    return model, optimizer, running_loss / len(train_loader)


def eval_model(model, device, val_ds):
    """Evaluate RNN model"""
    # Initialize running loss
    running_loss = np.zeros(len(model.loss))

    # Instantiate Data Loader for validation data
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=1)

    # Loop over validation data
    for X, y_true in val_loader:
        # Move data to GPU
        X = X.to(device)
        y_true = [y.to(device) for y in y_true]

        # Forward pass
        model(X)

        # Calculate loss
        loss = model.calc_loss(y_true)

        running_loss += [val.item() for val in loss]

    return running_loss / len(val_loader)


# Training loop
def training_loop(ds, model, device, nepochs=100, fname='',
                  optimizer_lr=0.001):
    """Training loop"""
    # Training and validation indices
    train_idx, val_idx = train_test_split(range(len(ds)), test_size=0.2)

    # Initialize minimum losses (the values are arbitrarily large)
    min_train_loss = 100
    min_val_loss = 100

    # Training and validation loss
    train_loss, val_loss = [], []

    # Move model to GPU
    model.to(device)

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)

    # Loop over epochs
    for i in range(nepochs):
        # Train one epoch
        model, optimizer, loss = train_one_epoch(model, device, optimizer,
                                                 Subset(ds, train_idx))
        train_loss.append(sum(loss))

        # Print losses
        print("================")
        for j, val in enumerate(model.loss.keys()):
            print(f"loss {model.loss[val].name} {val}: {loss[j]}")

        with torch.no_grad():
            # Change to evaluation mode
            model.eval()

            # Evaluate model
            loss = eval_model(model, device, Subset(ds, val_idx))
            val_loss.append(sum(loss))

        if train_loss[i] <= min_train_loss:
            # Save training checkpoints
            checkpoint = {"epoch": i + 1,
                          "model_state": model.state_dict(),
                          "optim_state": optimizer.state_dict(),
                          "loss": train_loss[i]}
            torch.save(checkpoint,
                       os.path.join("checkpoint", f"{fname}_train.pth"))

            # Update minimum train loss
            min_train_loss = train_loss[i]

        if val_loss[i] <= min_val_loss:
            # Save validation checkpoints
            checkpoint = {"epoch": i + 1,
                          "model_state": model.state_dict(),
                          "optim_state": optimizer.state_dict(),
                          "loss": val_loss[i]}
            torch.save(checkpoint,
                       os.path.join("checkpoint", f"{fname}_val.pth"))

            # Update minimum validation loss
            min_val_loss = val_loss[i]

        print(f"Step {i + 1}, Train loss {train_loss[i]}, "
              f"Val loss {val_loss[i]}")

        # Change to training mode
        model.train()

    return model, train_loss, val_loss
