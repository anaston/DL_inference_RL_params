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
        self.nagents = len(self.agent)

        # Number of blocks
        self.nblocks = self.df['block'].nunique()

        # Number of trials
        self.ntrials = self.df['trial'].nunique()

        # Number of bins for alpha and beta (remove NaNs!)
        try:
            self.alpha_nbins = self.df['alpha_bin'].dropna().nunique()
            self.beta_nbins = self.df['beta_bin'].dropna().nunique()
        except KeyError:  # for compatibility with old datasets
            self.alpha_nbins = 5
            self.beta_nbins = 5

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
        for i in vars:
            if i not in self.vars:
                # Initialize dictionary
                self.vars[i] = dict()

                # Set attributes
                attributes = {'encoding': encoding, 'dtype': dtype,
                              'shape': shape}
                for key in attributes:
                    for subkey, value in attributes[key].items():
                        if i in value:
                            self.vars[i][key] = subkey

        # Shape of the last dimension of variables
        for i in vars:
            if i in encoding['dummy']:
                # Dummy variable
                self.vars[i]['last_shape'] = np.unique(
                    self.df.loc[:, self.df.columns.str.startswith(i)].values).size
            else:
                # Other variable
                self.vars[i]['last_shape'] = 1

    def _var2tensor(self, df, varname):
        """Convert input/output variable to tensor"""
        # Get variable as numpy array
        if 'encoding' in self.vars[varname] \
                and self.vars[varname]['encoding'] == 'dummy':
            # Dummy variable
            var = np.zeros((len(df), self.vars[varname]['last_shape']))
            is_var = df.columns.str.startswith(varname)
            var[np.repeat(np.arange(len(df)), is_var.sum()), df.loc[
                :, is_var].values.flat] = 1
        else:
            # Other variable
            var = df[varname].values

        # Convert to tensor
        var = torch.tensor(var, dtype=self.vars[varname]['dtype'])

        # Reshape variable
        if 'shape' in self.vars[varname]:
            var = var.reshape(*self.vars[varname]['shape'])

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

        # Inputs - check if they are in dataframe
        self.inputs = inputs
        for i in inputs:
            if i not in self.df.columns:
                raise ValueError('Unknown input')

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

        # Outputs - check if they are in dataframe
        self.outputs = outputs
        for i in outputs:
            if i not in self.df.columns:
                raise ValueError('Unknown output')

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

        return X, y
