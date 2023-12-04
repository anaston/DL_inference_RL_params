# Import libraries
import numpy as np
import pandas as pd
from collections.abc import Iterator
from abc import abstractmethod


class Param:
    """
    Class that represents a parameter with a random value
    """
    def __init__(self, name, func, min=None, max=None, drift_rate=None,
                 nbins=5, **func_kwargs):
        # Parameter name
        self.name = name

        # Function for generating parameter
        self.func = func

        # Function keyword arguments
        self.func_kwargs = func_kwargs

        # Initialize random value for parameter
        self.value = self.func(**func_kwargs)

        # Minimum value for parameter
        if "low" in self.func_kwargs:
            self.min = func_kwargs["low"]
        else:
            self.min = min

        # Maximum value for parameter
        if "high" in self.func_kwargs:
            self.max = func_kwargs["high"]
        else:
            self.max = max

        # Drift rate for parameter
        self.drift_rate = drift_rate

        # Bins for parameter
        self.nbins = nbins

    def random(self):
        # Update random value for parameter
        self.value = self.func(**self.func_kwargs)

        return self.value

    def range(self):
        # Return range for parameter
        if self.min is None or self.max is None:
            raise ValueError("Min and max should be defined for range")
        else:
            return [self.min, self.max]

    def add_drift(self):
        if self.func == np.random.normal:
            self.value += self.func(0, self.drift_rate)
        else:
            raise ValueError("Only normal distribution is supported for drift")

    def transform(self):
        if self.func == np.random.normal:
            # Transform parameter
            if self.name == "alpha":
                return 1 / (1 + np.exp(-self.value))
            elif self.name == "beta":
                return np.exp(self.value).clip(min=self.min, max=self.max)
            else:
                raise ValueError("Only alpha and beta are supported for "
                                 "transformation")
        else:
            raise ValueError("Only normal distribution is supported for "
                             "transformation")


class ParamGeneratorStat(Iterator):
    """
    Class that represents a generator for stationary parameter
    """
    def __init__(self, name, func, ntrials=1000, **param_kwargs):
        # Instantiate a Parameter object
        self.param = Param(name, func, **param_kwargs)

        # Number of trials
        self.ntrials = ntrials

        # Starting index
        self._index = 0

    def __next__(self):
        """Iterate through trials"""
        if self._index < self.ntrials:
            # Increment index
            self._index += 1

            # Return stationary parameter value
            return self.param.value
        else:
            raise StopIteration


class ParamGeneratorNonStat(ParamGeneratorStat):
    """
    Class that represents a generator for a non-stationary parameter
    """
    def __init__(self, name, func, min_rep=100, prob_to_switch=0.005,
                 max_switch=np.random.choice([2, 3, 4]), **kwargs):
        """Initialize generator"""
        # Initialize parent class
        super().__init__(name, func, **kwargs)

        # Minimum repetition of parameter
        self.min_rep = min_rep

        # Maximum number of switches
        self.max_switch = max_switch

        # Probability to switch
        self.prob_to_switch = prob_to_switch

        # Number of repeating same value
        self.nrep = 0

        # Number of switches
        self.nswitch = 0

    def __next__(self):
        """Iterate through trials"""
        if self._index < self.ntrials:
            # Increment number of repeating same value
            self.nrep += 1

            # Update parameter value
            if (
                self.nrep > self.min_rep
                and self.nswitch < self.max_switch
                and np.random.random() < self.prob_to_switch
            ):
                self.param.random()
                self.nswitch += 1
                self.nrep = 0

            # Increment index
            self._index += 1

            # Return non-stationary parameter value
            return self.param.value
        else:
            raise StopIteration


class ParamGeneratorRandWalk(ParamGeneratorStat):
    """
    Class that represents a generator for a random-walk parameter
    """
    def __init__(self, name, func, drift_rate, **kwargs):
        """Initialize generator"""
        # Initialize parent class
        super().__init__(name, func, drift_rate=drift_rate, **kwargs)

    def __next__(self):
        """Iterate through trials"""
        if self._index < self.ntrials:
            if self._index > 0:
                self.param.add_drift()

            # Increment index
            self._index += 1

            # Return random-walk parameter value
            return self.param.transform()
        else:
            raise StopIteration


class QGenerator(Iterator):
    """
    Class that represents an abstract generator for a Q-learning agent
    """
    def __init__(self, alpha_gen, beta_gen, nactions, ntrials, block_size):
        """Initialize generator"""
        # Parameter generators
        self.alpha_gen = alpha_gen
        self.beta_gen = beta_gen

        # Initialize parameters
        self.alpha = None
        self.beta = None

        # Number of actions
        self.nactions = nactions

        # Initialize Q-values
        self.q = np.full(nactions, 0.5)

        # Initialize action and action probabilities
        self.action = None
        self.action_prob = np.zeros(nactions, dtype=float)

        # Initialize reward and reward probabilities
        self.reward = None
        self.reward_prob = np.zeros(nactions, dtype=float)

        # Number of trials
        self.ntrials = ntrials

        # Block size
        self.block_size = block_size

        # Starting index
        self._index = 0

    @abstractmethod
    def _choose_action(self):
        """Choose action"""
        pass

    @abstractmethod
    def _calc_reward(self):
        """Calculate reward"""
        pass

    def _update_q(self):
        """Update Q-value"""
        if (self._index + 1) % self.block_size == 0:
            # Reset Q-values at the end of a block
            self.q = np.full(self.nactions, 0.5)
        else:
            # Update Q-value for chosen action (Eq. 2 in Ger et al, 2023)
            self.q[self.action] = self.q[self.action] \
                + self.alpha * (self.reward - self.q[self.action])

    def __next__(self):
        """Iterate through trials"""
        if self._index < self.ntrials:
            # Generate parameters
            self.alpha = self.alpha_gen.__next__()
            self.beta = self.beta_gen.__next__()

            # Choose action
            self._choose_action()

            # Calculate reward
            self._calc_reward()

            # Update Q-value
            self._update_q()

            # Increment index
            self._index += 1
        else:
            raise StopIteration


class QGenerator2Armed(QGenerator):
    """
    Class that represents a generator for a Q-learning agent performing a
    2-armed bandit task with reward probabilities fixed within a block
    """
    def __init__(self, alpha_gen, beta_gen, nactions=2, **kwargs):
        """Initialize generator"""
        # Initialize parent class
        super().__init__(alpha_gen, beta_gen, nactions=nactions, **kwargs)

    def _choose_action(self):
        """Choose action"""
        # Calculate action probabilities (Eq. 3 in Ger et al, 2023)
        self.action_prob = np.exp(self.beta * self.q) \
            / np.sum(np.exp(self.beta * self.q))

        # Choose action
        self.action = np.random.choice(2, p=self.action_prob)

    def _calc_reward(self, prob=[0.1, 0.5, 0.9]):
        """Calculate reward"""
        # Randomly choose reward probability at the beginning of a block
        if self._index % self.block_size == 0:
            reward_prob = np.random.choice(prob)
            self.reward_prob = np.array([reward_prob, 1-reward_prob])

        # Calculate reward
        self.reward = np.random.choice(2, p=self.reward_prob)

    def __next__(self):
        # Call parent method
        super().__next__()

        # Return values
        # TODO: automate returned variables (e.g., to avoid hard-coding
        # variables or being able to define subset of variables)
        return (self.alpha, self.beta, self.q[0], self.q[1], self.action,
                self.action_prob[0], self.action_prob[1], self.reward,
                self.reward_prob[0], self.reward_prob[1])


class QGenerator4Armed(QGenerator):
    """
    Class that represents a generator for a Q-learning agent performing a
    4-armed bandit task
    """
    def __init__(self, alpha_gen, beta_gen, nactions=4, **kwargs):
        """Initialize generator"""
        # Initialize parent class
        super().__init__(alpha_gen, beta_gen, nactions=nactions, **kwargs)

        # Initialize offer
        self.offer = np.zeros(nactions, dtype=int)

        # Define offer set (i.e., all combinations of 2 actions)
        self.offer_set = np.stack(np.triu_indices(self.nactions, k=1), axis=-1)

    def _choose_action(self):
        """Choose action"""
        # Choose 2 actions (i.e. row) from offer set
        self.offer = self.offer_set[np.random.choice(self.offer_set.shape[0])]

        # Calculate action probabilities (Eq. 3 in Ger et al, 2023)
        self.action_prob = np.exp(self.beta * self.q[self.offer]) \
            / np.sum(np.exp(self.beta * self.q[self.offer]))

        # Choose action from offer
        self.action = np.random.choice(2, p=self.action_prob)
        self.action = self.offer[self.action]

    def _calc_reward(self, prob=[0.35, 0.45, 0.65, 0.75]):
        """Calculate reward"""
        # Randomly choose reward probability on each trial
        reward_prob = prob[self.action]
        self.reward_prob = np.array([reward_prob, 1-reward_prob])

        # Calculate reward
        self.reward = np.random.choice(2, p=self.reward_prob)

    def __next__(self):
        # Call parent method
        super().__next__()

        # Return values
        # TODO: automate returned variables (e.g., to avoid hard-coding
        # variables or being able to define subset of variables)
        return (self.alpha, self.beta, self.q[0], self.q[1], self.q[2],
                self.q[3], self.offer[0], self.offer[1], self.action,
                self.action_prob[0], self.action_prob[1], self.reward,
                self.reward_prob[0], self.reward_prob[1])


class QLearner:
    """
    Class that represents an abstract Q-learning agent
    """
    def __init__(self, idx_agent, q_gen, **q_gen_kwargs):
        """Initialize Q-learner"""
        # Agent index
        self.idx_agent = idx_agent

        # Generator for Q-learner
        self.q_gen = q_gen(**q_gen_kwargs)

        # Initialize dataframe
        self.df = pd.DataFrame(
            {"agent": np.full(self.q_gen.ntrials, self.idx_agent),
             "block": np.arange(self.q_gen.ntrials) // self.q_gen.block_size,
             "trial": np.arange(self.q_gen.ntrials)}
        )

    @abstractmethod
    def simulate(self):
        """Simulate agent"""
        pass

    def format_df(self, columns=["alpha", "beta", "action", "reward"]):
        """Format dataframe"""
        # Circular shift Q-values so that updated Q-values belong to next trial
        for i in range(self.q_gen.nactions):
            self.df[f'q{i+1}'] = np.roll(self.df[f'q{i+1}'], 1)

        # Convert parameter values to binned values
        if "alpha_bin" in columns:
            param = self.q_gen.alpha_gen.param
            self.df["alpha_bin"] = pd.cut(
                self.df["alpha"], bins=np.linspace(*param.range(), param.nbins+1),
                labels=np.arange(param.nbins)
            )
        if "beta_bin" in columns:
            param = self.q_gen.beta_gen.param
            self.df["beta_bin"] = pd.cut(
                self.df["beta"], bins=np.linspace(*param.range(), param.nbins+1),
                labels=np.arange(param.nbins)
            )

        # Return dataframe with selected columns
        return self.df[columns]


class QLearner2Armed(QLearner):
    """
    Class that represents a Q-learning agent performing a 2-armed bandit task
    """
    def __init__(self, idx_agent, **q_gen_kwargs):
        """Initialize Q-learner"""
        # Initialize parent class
        super().__init__(idx_agent, QGenerator2Armed, **q_gen_kwargs)

    def simulate(self):
        """Simulate agent performing a 2-armed bandit task"""
        # Loop through trials and add all results to dataframe
        self.df = self.df.join(pd.DataFrame([(
            alpha, beta, q1, q2, action, action_p1, action_p2, reward,
            reward_p1, reward_p2
        ) for (
            alpha, beta, q1, q2, action, action_p1, action_p2, reward,
            reward_p1, reward_p2
        ) in self.q_gen], columns=[
            'alpha', 'beta', 'q1', 'q2', 'action', 'action_prob1',
            'action_prob2', 'reward', 'reward_prob1', 'reward_prob2']))

        # Return self to encourage cascading
        return self


class QLearner4Armed(QLearner):
    """
    Class that represents a Q-learning agent performing a 4-armed bandit task
    """
    def __init__(self, idx_agent, **q_gen_kwargs):
        """Initialize Q-learner"""
        # Initialize parent class
        super().__init__(idx_agent, QGenerator4Armed, **q_gen_kwargs)

    def simulate(self):
        """Simulate agent performing a 4-armed bandit task"""
        # Loop through trials and add all results to dataframe
        self.df = self.df.join(pd.DataFrame([(
            alpha, beta, q1, q2, q3, q4, offer1, offer2, action, action_p1,
            action_p2, reward, reward_p1, reward_p2
        ) for (
            alpha, beta, q1, q2, q3, q4, offer1, offer2, action, action_p1,
            action_p2, reward, reward_p1, reward_p2
        ) in self.q_gen], columns=[
            'alpha', 'beta', 'q1', 'q2', 'q3', 'q4', 'offer1', 'offer2',
            'action', 'action_prob1', 'action_prob2', 'reward',
            'reward_prob1', 'reward_prob2']))

        # Return self to encourage cascading
        return self
