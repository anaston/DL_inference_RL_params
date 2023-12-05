# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from abc import abstractmethod


class BaseDataset(Dataset):
    """
    Class for abstract dataset of synthetic data

    NB: dataset is assumed to bo complete, i.e., all unique values are assumed
    to appear each variable
    """
    def __init__(self, df=None, path=None, alpha_nbins=None, beta_nbins=None,
                 backwards_compatible=False):
        """Initialize dataset"""
        # Dataset as dataframe
        if df is None:
            if path is None:
                raise ValueError('Either dataframe or path must be provided')
            else:
                self.df = pd.read_csv(path)
        else:
            self.df = df

        # Backwards compatibility
        self.backwards_compatible = backwards_compatible

        # Agents
        self.agent = self.df['agent'].unique()
        self.nagents = len(self.agent)

        # Number of blocks
        self.nblocks = self.df['block'].nunique()

        # Number of trials
        self.ntrials = self.df['trial'].nunique()

        # Number of bins for alpha and beta (remove NaNs!)
        # NB: all bins are assumed to be present in the dataset, otherwise
        # provide the number of bins as input
        if alpha_nbins is None:
            self.alpha_nbins = self.df['alpha_bin'].dropna().nunique()
        else:
            self.alpha_nbins = alpha_nbins
        if beta_nbins is None:
            self.beta_nbins = self.df['beta_bin'].dropna().nunique()
        else:
            self.beta_nbins = beta_nbins

        # Initialize variables
        self.vars = dict()

    @abstractmethod
    def __getitem__(self, idx):
        """Get agents as dataset items"""
        pass

    def __len__(self):
        """Define size of dataset as number of agents"""
        return self.agent.size

    def _set_vars(self, vars, encoding={'dummy': ['action', 'offer']},
                  dtype={torch.int64: ['alpha_bin', 'beta_bin'],
                         torch.float32: ['action', 'reward', 'offer', 'alpha', 'beta']},
                  shape={(-1, 1): ['alpha', 'beta', 'reward']}):
        """Set attributes of dataset variables"""
        # Index in dataframe
        for i in vars:
            if i not in self.vars:
                # Initialize dictionary
                self.vars[i] = dict()

            if i in encoding['dummy']:
                # Dummy variable
                self.vars[i]['idx'] = self.df.columns.str.startswith(i)
            else:
                # Other variable
                self.vars[i]['idx'] = self.df.columns == i
            if self.vars[i]['idx'].sum() == 0:
                raise ValueError(f'Variable {i} not found in dataset')

        # Set attributes such as encoding, dtype, and shape
        for i in vars:
            attributes = {'encoding': encoding, 'dtype': dtype,
                          'shape': shape}
            for key in attributes:
                for subkey, value in attributes[key].items():
                    if i in value:
                        self.vars[i][key] = subkey

        # Shape of the last dimension
        for i in vars:
            if i in encoding['dummy']:
                # Dummy variable
                self.vars[i]['last_shape'] = np.unique(
                    self.df.loc[:, self.vars[i]['idx']].values).size
            else:
                # Other variable
                self.vars[i]['last_shape'] = 1

    def _var2tensor(self, df, name):
        """Convert input/output variable to tensor"""
        # Get variable as numpy array
        if 'encoding' in self.vars[name] \
                and self.vars[name]['encoding'] == 'dummy':
            # Dummy variable
            var = np.zeros((len(df), self.vars[name]['last_shape']))
            var[np.repeat(np.arange(len(df)), self.vars[name]['idx'].sum()),
                df.loc[:, self.vars[name]['idx']].values.flat] = 1
        else:
            # Other variable
            var = df[name].values

        # Convert to tensor
        var = torch.tensor(var, dtype=self.vars[name]['dtype'])

        # Reshape variable
        if 'shape' in self.vars[name]:
            var = var.reshape(*self.vars[name]['shape'])

        # Return variable as tensor
        return var


class UnlabeledDataset(BaseDataset):
    """
    Unlabeled dataset of synthetic data
    """
    def __init__(self, inputs, **kwargs):
        """Initialize dataset"""
        # Initialize parent class
        super().__init__(**kwargs)

        # Inputs
        self.inputs = inputs

        # Set attributes of inputs
        self._set_vars(inputs)

        # Number of inputs
        self.ninputs = sum([self.vars[i]['last_shape'] for i in inputs])

    def __getitem__(self, idx):
        """Get agents as dataset items"""
        # Get dataframe for list of agents
        df = self.df[self.df['agent'] == self.agent[idx]]

        # Convert inputs to tensors and stack them
        X = torch.hstack([self._var2tensor(df, i) for i in self.inputs])

        # Add shift to input data
        # NB: if compatibility needed with old datasets
        if self.backwards_compatible:
            X = nn.functional.pad(X, [0, 0, 1, 0], 'constant', value=0)[:-1]

        # Add dummy zeros to the beginning of each block
        X.reshape(self.nblocks, -1, X.shape[1])[:, 0, :] = (
            torch.zeros(size=(self.nblocks, X.shape[1]))
        )

        # Return input data
        return X


class LabeledDataset(UnlabeledDataset):
    """
    Labeled dataset of synthetic data
    """
    def __init__(self, inputs, outputs, **kwargs):
        """Initialize dataset"""
        # Initialize parent class
        super().__init__(inputs, **kwargs)

        # Outputs
        self.outputs = outputs

        # Set attributes of outputs
        self._set_vars(outputs)

    def __getitem__(self, idx):
        """Get agents as dataset items"""
        # Call parent method
        X = super().__getitem__(idx)

        # Get dataframe for agent
        df = self.df[self.df['agent'] == self.agent[idx]]

        # Convert outputs to tensors
        # NB: we keep outputs in list to be able to use specific dtype for each
        y = [self._var2tensor(df, o) for o in self.outputs]

        # Add shift to action data so that it corresponds to the next trial
        # NB: avoid if compatibility needed with old datasets
        if not self.backwards_compatible:
            y = [nn.functional.pad(val, [0, 0, 0, 1], 'constant', value=0)[1:]
                 if self.outputs[i] == 'action' else val
                 for i, val in enumerate(y)]

        return X, y
