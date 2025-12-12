"""
Agent class for Trade or Tighten game.

TODO: POMDP extension - add belief state tracking here
TODO: Inverse planning - add methods for belief inference and updating
"""

class Agent:
    """
    Represents a trader agent in the market game.

    Attributes:
        agent_id: identifier (0, 1, 2 for A, B, C)
        name: string name ("A", "B", "C")
        mu_mean: mean of prior belief distribution
        sigma_mu: std dev of prior belief distribution
        sigma: confidence parameter (fixed)
        beta: rationality parameter for softmax policy
        mu_i: sampled private belief (set after initialization via Memo)
    """

    def __init__(self, agent_id, name, mu_mean, sigma_mu, sigma, beta):
        self.agent_id = agent_id
        self.name = name
        self.mu_mean = mu_mean
        self.sigma_mu = sigma_mu
        self.sigma = sigma
        self.beta = beta
        self.mu_i = None  # will be set by Memo when beliefs are sampled

    def get_expected_value(self):
        """
        Return μ_i (for reward calculation).

        Returns:
            float: The agent's private belief about asset value
        """
        if self.mu_i is None:
            raise ValueError("mu_i has not been set. Beliefs must be sampled via Memo first.")
        return self.mu_i

    def __repr__(self):
        return f"Agent(id={self.agent_id}, name={self.name}, mu_mean={self.mu_mean}, beta={self.beta})"
