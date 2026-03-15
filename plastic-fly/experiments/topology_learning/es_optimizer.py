"""
OpenAI-style Evolution Strategy optimizer.

Reference: Salimans et al. 2017, "Evolution Strategies as a Scalable
Alternative to Reinforcement Learning"

Features:
  - Antithetic sampling (mirrored perturbations)
  - Rank-based fitness shaping
  - Weight decay
  - Seed-based noise: stores only a random seed, regenerates noise
    vectors one at a time in both ask() and tell(). Memory usage is
    O(params) instead of O(pop * params).
  - float32 throughout to halve memory footprint.
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
        self.theta = np.zeros(n_params, dtype=np.float32)

        # Noise seed for current generation (regenerated each ask/tell cycle)
        self._seed = 0

        # Generation counter
        self.generation = 0

    def _iter_noise(self):
        """Yield noise vectors one at a time from the stored seed."""
        rng = np.random.RandomState(self._seed)
        half = self.pop_size // 2 if self.antithetic else self.pop_size
        for _ in range(half):
            yield rng.randn(self.n_params).astype(np.float32)

    def ask(self) -> list[np.ndarray]:
        """Generate population of parameter vectors to evaluate.

        Returns list of (pop_size,) parameter vectors.
        With antithetic sampling, generates pop_size/2 noise vectors
        and evaluates both +noise and -noise.
        """
        self._seed = np.random.randint(0, 2**31)
        population = []
        if self.antithetic:
            for noise in self._iter_noise():
                population.append(self.theta + self.sigma * noise)
                population.append(self.theta - self.sigma * noise)
        else:
            for noise in self._iter_noise():
                population.append(self.theta + self.sigma * noise)
        return population

    def tell(self, rewards: list[float]):
        """Update parameters using fitness-shaped rewards.

        Args:
            rewards: list of floats, one per population member from ask().
        """
        rewards = np.array(rewards, dtype=np.float32)

        # Rank-based fitness shaping (more robust than raw rewards)
        ranks = np.zeros_like(rewards)
        order = rewards.argsort()
        for i, idx in enumerate(order):
            ranks[idx] = i
        # Normalize to [-0.5, 0.5]
        shaped = (ranks / max(len(ranks) - 1, 1)) - 0.5

        # Compute gradient estimate by regenerating noise from the same seed
        grad = np.zeros(self.n_params, dtype=np.float32)
        if self.antithetic:
            half = self.pop_size // 2
            for i, noise in enumerate(self._iter_noise()):
                grad += (shaped[2 * i] - shaped[2 * i + 1]) * noise
            grad /= (2 * half * self.sigma)
        else:
            for i, noise in enumerate(self._iter_noise()):
                grad += shaped[i] * noise
            grad /= (self.pop_size * self.sigma)

        # Update with weight decay
        self.theta += self.lr * grad - self.lr * self.weight_decay * self.theta
        self.generation += 1

    def set_params(self, theta: np.ndarray):
        """Set the current mean parameter vector."""
        self.theta = theta.astype(np.float32).copy()

    @property
    def best_params(self) -> np.ndarray:
        """Current mean parameters (the 'best' estimate)."""
        return self.theta.copy()
