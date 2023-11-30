# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class UnlabeledDataset(Dataset):
    """
    Unlabeled dataset of synthetic data

    NB: dtype of tensors is essential for the model to work later on
    """
    def __init__(self, df=None, path=None):
        """Initialize dataset"""
        # Dataset as dataframe
        if df is None:
            if path is None:
                raise ValueError('Either dataframe or path must be provided')
            else:
                self.df = pd.read_csv(path)
        else:
            self.df = df

        # Agents
        self.agent = self.df['agent'].unique()

        # Number of actions
        self.nactions = self.df['action'].nunique()

    def __getitem__(self, idx):
        """Get agents as dataset items"""
        # Get dataframe for agent
        df = self.df[self.df['agent'] == self.agent[idx]]

        # Number of blocks
        nblocks = df['block'].nunique()

        # Convert action to one-hot encoded tensor
        action_onehot = nn.functional.one_hot(
            torch.from_numpy(df['action'].values),
            self.nactions).type(dtype=torch.float32)

        # Reward as tensor
        reward = torch.tensor(df['reward'].values,
                              dtype=torch.float32)[:, np.newaxis]

        # Input data (with dummy zeros in the beginning)
        # TODO: check if first step is necessary as there appears to be
        # block-wise padding in the second step
        X = nn.functional.pad(torch.hstack((reward, action_onehot)),
                              [0, 0, 1, 0], 'constant', value=0)[:-1]
        X.reshape(nblocks, -1, X.shape[1])[:, 0, :] = (
            torch.zeros(size=(nblocks, X.shape[1]))
        )

        # Return input data
        return X

    def __len__(self):
        """Define size of dataset as number of agents"""
        return len(self.agent)


class LabeledDataset(UnlabeledDataset):
    """
    Labeled dataset of synthetic data
    """
    def __init__(self, labels, df=None, path=None):
        """Initialize dataset"""
        # Initialize parent class
        super().__init__(df=df, path=path)

        # Labels
        self.labels = labels

        # Number of bins for alpha and beta
        if 'alpha_bin' in labels:
            self.nbins_alpha = self.df['alpha_bin'].nunique()
        if 'beta_bin' in labels:
            self.nbins_beta = self.df['beta_bin'].nunique()

    def __getitem__(self, idx):
        """Get agents as dataset items"""
        # Call parent method
        X = super().__getitem__(idx)

        # Get dataframe for agent
        df = self.df[self.df['agent'] == self.agent[idx]]

        # Multi-output data
        # NB: tensors must be in specific shapes to match those of model
        # outputs and/or requirements for loss functions in rnn.multi_loss
        # It is also critical that the order of labels must match the
        # order in rnn.multi_loss
        y = []
        for label in self.labels:
            if label == 'action':
                if df['action'].nunique() == 1:
                    y.append(torch.from_numpy(
                        np.repeat(df['action'].values, self.nactions).reshape(
                            -1, self.nactions).type(dtype=torch.float32)))
                else:
                    y.append(nn.functional.one_hot(
                        torch.from_numpy(df['action'].values),
                        self.nactions).type(dtype=torch.float32),)
            elif label in ['alpha_bin', 'beta_bin']:
                y.append(torch.tensor(df[label].values, dtype=torch.int64))
            elif label in ['alpha', 'beta']:
                y.append(torch.tensor(df[label].values,
                                      dtype=torch.float32).reshape(-1, 1))
            else:
                raise ValueError('Unknown label')
        
        return X, y

    def __len__(self):
        """Define size of dataset as number of agents"""
        return len(self.agent)
