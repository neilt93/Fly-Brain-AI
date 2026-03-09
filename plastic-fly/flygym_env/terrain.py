"""
Custom terrain: flat ground followed by blocks.

The fly spawns on the flat section and walks forward into blocks,
experiencing the terrain change mid-episode without simulation restart.
"""

import numpy as np
from typing import Optional
from flygym.arena.base import BaseArena


class FlatThenBlocksTerrain(BaseArena):
    """Flat terrain for the first `flat_length` mm, then blocks terrain.

    The fly spawns at x=0 on the flat section. As it walks forward (+x),
    it encounters blocks starting at x=flat_length.

    Parameters
    ----------
    flat_length : float
        Length of the flat section in mm (x-axis), by default 8.
    blocks_length : float
        Length of the blocks section in mm, by default 15.
    block_size : float
        Side length of each block in mm, by default 1.3.
    height_range : tuple[float, float]
        Min/max height of raised blocks in mm, by default (0.2, 0.4).
    y_range : tuple[float, float]
        Width of the terrain in mm, by default (-20, 20).
    friction : tuple[float, float, float]
        Sliding, torsional, and rolling friction, by default (1, 0.005, 0.0001).
    ground_alpha : float
        Opacity of the ground, by default 1.0.
    rand_seed : int
        Seed for block height randomization, by default 0.
    """

    def __init__(
        self,
        flat_length: float = 8.0,
        blocks_length: float = 15.0,
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.2, 0.4),
        y_range: tuple[float, float] = (-20, 20),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
    ):
        super().__init__()
        self.friction = friction
        self.flat_length = flat_length
        self.blocks_start = flat_length
        self.blocks_end = flat_length + blocks_length
        self._max_block_height = 0.0
        self._height_expected = np.mean(height_range)

        rand_state = np.random.RandomState(rand_seed)

        # --- Flat section: x in [-2, flat_length] ---
        flat_x_start = -2.0
        flat_x_size = (flat_length - flat_x_start) / 2
        self.root_element.worldbody.add(
            "geom",
            type="box",
            name="ground_flat",
            size=(flat_x_size, (y_range[1] - y_range[0]) / 2, 1.0),
            pos=((flat_x_start + flat_length) / 2, 0, -1.0),
            friction=friction,
            rgba=(0.35, 0.35, 0.35, ground_alpha),
        )

        # --- Blocks section: x in [flat_length, flat_length + blocks_length] ---
        blocks_x_start = flat_length
        blocks_x_end = flat_length + blocks_length

        x_centers = np.arange(
            blocks_x_start + block_size / 2, blocks_x_end, block_size
        )
        y_centers = np.arange(
            y_range[0] + block_size / 2, y_range[1], block_size
        )

        for i, x_pos in enumerate(x_centers):
            for j, y_pos in enumerate(y_centers):
                # Checkerboard: only raise diagonal blocks
                if (i + j) % 2 == 0:
                    height = 0.1 + rand_state.uniform(*height_range)
                else:
                    height = 0.1

                box_size = (
                    block_size / 2 * 1.1,
                    block_size / 2 * 1.1,
                    height / 2 + block_size / 2,
                )
                box_pos = (
                    x_pos,
                    y_pos,
                    height / 2 - block_size / 2 - self._height_expected - 0.1,
                )
                self._max_block_height = max(
                    self._max_block_height,
                    height - self._height_expected - 0.1,
                )
                self.root_element.worldbody.add(
                    "geom",
                    type="box",
                    name=f"block_x{i}_y{j}",
                    size=box_size,
                    pos=box_pos,
                    friction=friction,
                    rgba=(0.25, 0.25, 0.3, ground_alpha),
                )

        # Base plane under blocks (catch falls)
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground_blocks_base",
            pos=((blocks_x_start + blocks_x_end) / 2, 0, -2.0),
            size=((blocks_x_end - blocks_x_start) / 2, (y_range[1] - y_range[0]) / 2, 1),
            friction=friction,
            rgba=(0.2, 0.2, 0.2, ground_alpha),
        )

        # --- Flat section after blocks (for forgetting test) ---
        post_start = blocks_x_end
        post_end = blocks_x_end + 10.0
        self.root_element.worldbody.add(
            "geom",
            type="box",
            name="ground_flat_post",
            size=((post_end - post_start) / 2, (y_range[1] - y_range[0]) / 2, 1.0),
            pos=((post_start + post_end) / 2, 0, -1.0),
            friction=friction,
            rgba=(0.35, 0.35, 0.35, ground_alpha),
        )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def _get_max_floor_height(self) -> float:
        return max(0, self._max_block_height)
