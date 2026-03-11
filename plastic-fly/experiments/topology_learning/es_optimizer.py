"""
OpenAI-style Evolution Strategy optimizer.

Reference: Salimans et al. 2017, "Evolution Strategies as a Scalable
Alternative to Reinforcement Learning"

Features:
  - Antithetic sampling (mirrored perturbations)
  - Rank-based fitness shaping
  - Weight decay
"""

import numpy as np


class OpenAIES:
    """Evolution Strategy for optimizing policy parameters."""

    def __init__(
        self,
        n_params: int,
        sigma: float = 0.02,
        lr: float = 0.01,
        pop_size: int = 32,
        antithetic: bool = True,
        weight_decay: float = 0.005,
    ):
        self.n_params = n_params
        self.sigma = sigma
        self.lr = lr
        self.pop_size = pop_size
        self.antithetic = antithetic
        self.weight_decay = weight_decay

        # Current parameter vector (mean of search distribution)
        self.theta = np.zeros(n_params, dtype=np.float64)

        # Generation counter
        self.generation = 0

    def ask(self) -> list[np.ndarray]:
        """Generate population of parameter vectors to evaluate.

        Returns list of (pop_size,) parameter vectors.
        With antithetic sampling, generates pop_size/2 noise vectors
        and evaluates both +noise and -noise.
        """
        if self.antithetic:
            half = self.pop_size // 2
            self._noise = np.random.randn(half, self.n_params).astype(np.float64)
            population = []
            for i in range(half):
                population.append(self.theta + self.sigma * self._noise[i])
                population.append(self.theta - self.sigma * self._noise[i])
        else:
            self._noise = np.random.randn(self.pop_size, self.n_params).astype(np.float64)
            population = [
                self.theta + self.sigma * self._noise[i]
                for i in range(self.pop_size)
            ]
        return population

    def tell(self, rewards: list[float]):
        """Update parameters using fitness-shaped rewards.

        Args:
            rewards: list of floats, one per population member from ask().
        """
        rewards = np.array(rewards, dtype=np.float64)

        # Rank-based fitness shaping (more robust than raw rewards)
        ranks = np.zeros_like(rewards)
        order = rewards.argsort()
        for i, idx in enumerate(order):
            ranks[idx] = i
        # Normalize to [-0.5, 0.5]
        shaped = (ranks / max(len(ranks) - 1, 1)) - 0.5

        # Compute gradient estimate
        if self.antithetic:
            half = self.pop_size // 2
            grad = np.zeros(self.n_params, dtype=np.float64)
            for i in range(half):
                grad += (shaped[2 * i] - shaped[2 * i + 1]) * self._noise[i]
            grad /= (2 * half * self.sigma)
        else:
            grad = np.zeros(self.n_params, dtype=np.float64)
            for i in range(self.pop_size):
                grad += shaped[i] * self._noise[i]
            grad /= (self.pop_size * self.sigma)

        # Update with weight decay
        self.theta += self.lr * grad - self.lr * self.weight_decay * self.theta
        self.generation += 1

    def set_params(self, theta: np.ndarray):
        """Set the current mean parameter vector."""
        self.theta = theta.copy()

    @property
    def best_params(self) -> np.ndarray:
        """Current mean parameters (the 'best' estimate)."""
        return self.theta.copy()
