import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class GRU(nn.Module):
    """
    Class for custom RNN model with two GRU layers
    """

    def __init__(self, input_size, hidden_size, alpha_embedding_size,
                 beta_embedding_size, output_size, dropout,):
        """Initialize the model"""
        # Initialize parent class
        super(GRU, self).__init__()

        # Set size attributes
        self.input_size = input_size
        self.alpha_embedding_size = alpha_embedding_size
        self.beta_embedding_size = beta_embedding_size

        # GRU layer on inputs
        self.gru_input = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                num_layers=1, batch_first=True,
                                dropout=dropout,)

        # Linear layers on GRU output
        self.linear_alpha_bin = nn.Linear(hidden_size, alpha_embedding_size)
        self.linear_beta_bin = nn.Linear(hidden_size, beta_embedding_size)

        # RELU layers on linear output
        self.relu_alpha = nn.ReLU()
        self.relu_beta = nn.ReLU()

        # Linear layers on RELU output
        self.linear_alpha = nn.Linear(alpha_embedding_size, 1)
        self.linear_beta = nn.Linear(beta_embedding_size, 1)

        # GRU layer on embedded inputs
        self.gru_embedded = nn.GRU(input_size=input_size
                                   + alpha_embedding_size
                                   + beta_embedding_size,
                                   hidden_size=hidden_size, num_layers=1,
                                   batch_first=True, dropout=dropout,)

        # Linear layer on embedded GRU output
        self.linear_action = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        """Forward pass of the model"""
        # Predict hidden estimates using GRU layer
        y_hidden, state_hidden = self.gru_input(X)

        # Predict binary alpha and beta estimates using linear and RELU layers
        y_alpha_bin = self.relu_alpha(self.linear_alpha_bin(y_hidden))
        y_beta_bin = self.relu_beta(self.linear_beta_bin(y_hidden))

        # Predict alpha and beta estimates using linear layer
        y_alpha = self.linear_alpha(y_alpha_bin)
        y_beta = self.linear_beta(y_beta_bin)

        # Create embedded input
        X_embedded = torch.cat((X, y_alpha_bin, y_beta_bin), dim=2)

        # Predict action using GRU layer
        y_action, state_action = self.gru_embedded(X_embedded)
        y_action = F.softmax(self.linear_action(y_action), dim=-1)

        return (y_action, y_alpha_bin, y_beta_bin, y_alpha, y_beta,
                state_hidden, state_action,)


def multi_loss(y_action, y_alpha_bin, y_beta_bin, y_alpha, y_beta):
    """Multi-loss function"""
    # Compute individual losses
    # TODO: check dimensions and order of inputs
    loss = [nn.BCELoss()(*y_action),
            nn.CrossEntropyLoss()(*y_alpha_bin),
            nn.CrossEntropyLoss()(*y_beta_bin),
            nn.MSELoss()(*y_alpha),
            nn.MSELoss()(*y_beta)]

    # Return weighted losses
    loss_weights = [1, 0.2, 0.2, 6, 0.1]
    return [loss_weights[i] * val for i, val in enumerate(loss)]


def train_one_epoch(model, device, optimizer, train_loader):
    """Train RNN model for one epoch"""
    # Initialize running loss as array
    running_loss = np.zeros(5)

    # Loop over training data
    for X, y_true in train_loader:
        # Move data to GPU
        X = X.to(device)
        y_true = [y.to(device) for y in y_true]

        # Zero the gradient buffers
        optimizer.zero_grad()

        # Forward pass
        y_action, y_alpha_bin, y_beta_bin, y_alpha, y_beta, _, _ = model(X)

        # Compute loss
        # NB: input dimensions need to be transposed for CE loss
        # TODO: automate the order of y_true
        loss = multi_loss((y_action, y_true[0]),
                          (torch.transpose(y_alpha_bin, 1, 2), y_true[1]),
                          (torch.transpose(y_beta_bin, 1, 2), y_true[2]),
                          (y_alpha, y_true[3]),
                          (y_beta, y_true[4]),)

        # Backpropagate loss
        sum(loss).backward()

        # Update RNN weights
        optimizer.step()

        # Update loss
        running_loss += [val.item() for val in loss]

    return model, optimizer, running_loss / len(train_loader)


def eval_model(model, device, val_loader):
    """Evaluate RNN model"""
    # Initialize running loss
    running_loss = np.zeros(5)
    
    # Loop over validation data
    for X, y_true in val_loader:
        # Move data to GPU
        X = X.to(device)
        y_true = [y.to(device) for y in y_true]

        # Forward pass
        y_action, y_alpha_bin, y_beta_bin, y_alpha, y_beta, _, _ = model(X)

        # Compute loss
        # NB: input dimensions need to be transposed for CE loss
        loss = multi_loss((y_action, y_true[0]),
                          (torch.transpose(y_alpha_bin, 1, 2), y_true[1]),
                          (torch.transpose(y_beta_bin, 1, 2), y_true[2]),
                          (y_alpha, y_true[3]),
                          (y_beta, y_true[4]),)
        running_loss += [val.item() for val in loss]

    return running_loss / len(val_loader)


# Training loop
def training_loop(model, device, train_loader, val_loader,
                  fname, nepochs=100):
    """Training loop"""
    min_train_loss = 100
    min_val_loss = 100

    # Training and validation loss
    train_loss, val_loss = [], []

    # Move model to GPU
    model.to(device)

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Loop over epochs
    for i in range(nepochs):
        # Train one epoch
        model, optimizer, loss = train_one_epoch(model, device, optimizer,
                                                 train_loader)
        train_loss.append(sum(loss))

        # Print losses
        print("================")
        loss_names = ["BCE action", "CE alpha", "CE beta", "MSE alpha",
                      "MSE beta"]
        for j, val in enumerate(loss):
            print(f"loss {loss_names[j]}: {val}")

        with torch.no_grad():
            # Change to evaluation mode
            model.eval()

            # Evaluate model
            loss = eval_model(model, device, val_loader)
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
