import numpy as np
import pandas as pd


class Param:
    """
    Class that represents free parameters for RL models
    """

    def __init__(self, name, func=np.random.uniform, func_kwargs=None,
                 nbins=5,):
        # Parameter name
        self.name = name

        # Function for generating parameter
        self.func = func

        # Function keyword arguments
        self.func_kwargs = func_kwargs

        # Bins for parameter
        self.nbins = nbins

    def random(self):
        # Return random value for parameter
        return self.func(**self.func_kwargs)

    def range(self):
        # Return range for parameter if low and high are defined
        if "low" not in self.func_kwargs or "high" not in self.func_kwargs:
            raise ValueError("Low and high must be defined for range")
        else:
            return self.func_kwargs["low"], self.func_kwargs["high"]


class QLearnerStationary:
    """
    Class that represents a stationary Q-learning agent
    """

    def __init__(self, idx_agent, params, nactions=2, ntrials=1000,
                 block_size=100,):
        """Initialize Q-learner"""
        # Free RL model parameters: alpha and beta
        if "alpha" not in params or "beta" not in params:
            raise ValueError("At least one free parameter must be specified")
        else:
            self.params = {k: Param(k, **val) for k, val in params.items()}

        # Agent index
        self.idx_agent = idx_agent

        # Number of actions
        self.nactions = nactions

        # Number of trials
        self.ntrials = ntrials

        # Number of trials per block
        self.block_size = block_size

        # Initialize free parameters of the RL model
        self.alpha = np.full(self.ntrials, self.params["alpha"].random())
        self.beta = np.full(self.ntrials, self.params["beta"].random())

        # Calculate expected reward
        self.exp_reward = self._calc_exp_reward()

        # Initialize Q-values and related action probabilities
        self.q = np.zeros((self.ntrials, 2), dtype=float)
        self.action_prob = np.zeros((self.ntrials, 2), dtype=float)

        # Initialize action and reward
        self.action = np.zeros(self.ntrials, dtype=int)
        self.reward = np.zeros(self.ntrials, dtype=int)

    def _calc_exp_reward(self, prob=[0.1, 0.5, 0.9]):
        """Calculate expected reward"""
        exp_reward = np.concatenate(
            [np.full(self.block_size, np.random.choice(prob))
             for _ in range(self.ntrials // self.block_size)]
        )

        return np.stack((exp_reward, 1 - exp_reward), axis=1)

    def _qlearner(self, q, exp_reward, alpha, beta):
        """Q-learner"""
        # Calculate action probabilities
        action_prob = np.exp(beta * q) / np.sum(np.exp(beta * q))

        # Choose action
        action = np.random.choice([0, 1], p=action_prob)

        # Calculate reward
        reward = np.random.choice([0, 1], p=[exp_reward[1 - action],
                                             exp_reward[action]])

        # Calculate prediction error
        pred_error = reward - q[action]

        # Update q value for chosen action
        q[action] = q[action] + alpha * pred_error

        return q, action_prob, action, reward

    def simulate(self):
        """Simulate agent"""
        # Loop through trials
        for t in range(self.ntrials):
            # Update Q-values
            if t % self.block_size > 0:
                self.q[t, :] = self.q[t - 1, :]

            # Run Q-learner
            self.q[t, :], self.action_prob[t, :], self.action[t], self.reward[t] = \
                self._qlearner(self.q[t, :], self.exp_reward[t, :], self.alpha[t], self.beta[t])

    def to_df(self, columns=["alpha", "beta", "action", "reward"]):
        """Create dataframe from qlearner inputs and outputs"""
        # Initialize dataframe
        df = pd.DataFrame({"agent": np.full(self.ntrials, self.idx_agent),
                           "block": np.arange(self.ntrials) // self.block_size,
                           "trial": np.arange(self.ntrials), }, )

        # Add columns to dataframe
        for col in columns:
            if "bin" in col:
                # Convert parameters values to binned values
                param_name = col.replace("_bin", "")
                param = self.params[param_name]
                df[col] = pd.cut(getattr(self, param_name),
                                 bins=np.linspace(*param.range(), param.nbins+1),
                                 labels=np.arange(param.nbins),)
            elif getattr(self, col).ndim == 1:
                # 1D variables
                df[col] = getattr(self, col)
            else:
                # >=2D variables
                for i in range(getattr(self, col).shape[1]):
                    df[f"{col}_{i}"] = getattr(self, col)[:, i]

        return df


class QLearnerNonStationary(QLearnerStationary):
    """
    Class that represents a non-stationary Q-learning agent
    """

    def __init__(self, idx_agent, params, max_cntr=100, max_switch=np.random.choice([2, 3, 4]),
                 prob_to_switch=0.005, **kwargs):
        """Initialize Q-learner"""
        super().__init__(idx_agent, params, **kwargs)

        # Maximum number for counter
        self.max_cntr = max_cntr

        # Maximum number of switches
        self.max_switch = max_switch

        # Probability to switch
        self.prob_to_switch = prob_to_switch

    def simulate(self):
        """Simulate agent"""
        # Loop through trials
        for t in range(self.ntrials):
            # Update Q-values
            if t % self.block_size > 0:
                self.q[t, :] = self.q[t - 1, :]

            # Counters for alpha, beta and their switches
            c_a, c_as, c_b, c_bs = 0, 0, 0, 0

            # Update alpha
            if c_a > self.max_cntr and c_as < self.max_switch and np.random.random() < self.prob_to_switch:
                self.alpha[t] = self.params["alpha"].random()
                c_as += 1
                c_a = 0

            # Update alpha
            if c_b > self.max_cntr and c_bs < self.max_switch and np.random.random() < self.prob_to_switch:
                self.beta[t] = self.params["beta"].random()
                c_bs += 1
                c_b = 0

            # Run Q-learner
            self.q[t, :], self.action_prob[t, :], self.action[t], self.reward[t] = \
                self._qlearner(self.q[t, :], self.exp_reward[t, :], self.alpha[t], self.beta[t])


class QLearnerRandomWalk(QLearnerStationary):
    """
    Class that represents a random-walk Q-learning agent
    """

    def __init__(self, idx_agent, params, drift_rate, **kwargs):
        """Initialize Q-learner"""
        super().__init__(idx_agent, params, **kwargs)

        # Drift rate
        self.drift_rate = drift_rate

    def simulate(self):
        """Simulate agent"""
        # Loop through trials
        for t in range(self.ntrials):
            # Update Q-values
            if t % self.block_size > 0:
                self.q[t, :] = self.q[t - 1, :]

            # Update alpha and beta
            if t > 0:
                self.alpha[t] += np.random.normal(0, self.drift_rate['alpha'])
                self.beta[t] += np.random.normal(0, self.drift_rate['beta'])

            # Transform beta
            self.beta[t] = np.exp(self.beta[t]).clip(0, 10)

            # Transform alpha
            self.alpha[t] = 1 / (1 + np.exp(-self.alpha[t]))

            # Run Q-learner
            self.q[t, :], self.action_prob[t, :], self.action[t], self.reward[t] = \
                self._qlearner(self.q[t, :], self.exp_reward[t, :], self.alpha[t], self.beta[t])