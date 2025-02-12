from __future__ import annotations
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from gym.spaces import Box
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.core.constants import COLOR_NAMES
import itertools as itt


# Original: /home/smirnov/miniconda3/envs/fedrl/lib/python3.11/site-packages/minigrid/envs/crossing.py
class CustomCrossingEnv(MiniGridEnv):
    """
    ## Description

    Depending on the `obstacle_type` parameter:
    - `Lava` - The agent has to reach the green goal square on the other corner
        of the room while avoiding rivers of deadly lava which terminate the
        episode in failure. Each lava stream runs across the room either
        horizontally or vertically, and has a single crossing point which can be
        safely used; Luckily, a path to the goal is guaranteed to exist. This
        environment is useful for studying safety and safe exploration.
    - otherwise - Similar to the `LavaCrossing` environment, the agent has to
        reach the green goal square on the other corner of the room, however
        lava is replaced by walls. This MDP is therefore much easier and maybe
        useful for quickly testing your algorithms.

    ## Mission Space
    Depending on the `obstacle_type` parameter:
    - `Lava` - "avoid the lava and get to the green goal square"
    - otherwise - "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of the map SxS.
    N: number of valid crossings across lava or walls from the starting position
    to the goal

    - `Lava` :
        - `MiniGrid-LavaCrossingS9N1-v0`
        - `MiniGrid-LavaCrossingS9N2-v0`
        - `MiniGrid-LavaCrossingS9N3-v0`
        - `MiniGrid-LavaCrossingS11N5-v0`

    - otherwise :
        - `MiniGrid-SimpleCrossingS9N1-v0`
        - `MiniGrid-SimpleCrossingS9N2-v0`
        - `MiniGrid-SimpleCrossingS9N3-v0`
        - `MiniGrid-SimpleCrossingS11N5-v0`

    """

    def __init__(
        self,
        size=9,
        num_crossings=1,
        obstacles_dir: int = 1, # [0: vertical and horizontal, 1: vertical, 2: horizontal]
        see_through_walls=True,
        obstacle_type=Lava,
        max_steps: int | None = None,
        agent_corner: int = 0, # from [0, 1, 2, 3]
        goal_corner: int = 2, # from [0, 1, 2, 3]
        **kwargs,
    ):
        self.obstacles_dir = obstacles_dir
        # self.number_of_generated_grids = 0
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        self.goal_position = None
        self.agent_corner = agent_corner
        self.goal_corner = goal_corner

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=see_through_walls,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        if self.agent_corner == 0:
            self.agent_pos = np.array((1, 1))
        elif self.agent_corner == 1:
            self.agent_pos = np.array((width - 2, 1))
        elif self.agent_corner == 2:
            self.agent_pos = np.array((width - 2, height - 2))
        elif self.agent_corner == 3:
            self.agent_pos = np.array((1, height - 2))
        else:
            assert False

        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        x, y = None, None
        if self.goal_corner == 0:
            x, y = 1, 1
        elif self.goal_corner == 1:
            x, y = width - 2, 1
        elif self.goal_corner == 2:
            x, y = width - 2, height - 2
        elif self.goal_corner == 3:
            x, y = 1, height - 2
        else:
            assert False

        self.put_obj(Goal(), x, y)
        self.goal_position = (x, y)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        if self.obstacles_dir == 0:
            rivers = [
                (v, i) for i in range(2, height - 2, 2)
            ] + [
                (h, j) for j in range(2, width - 2, 2)
            ]
        elif self.obstacles_dir == 1:
            rivers = [(v, i) for i in range(2, height - 2, 2)]
        elif self.obstacles_dir == 2:
            rivers = [(h, j) for j in range(2, width - 2, 2)]
        else:
            assert False

        self.np_random.shuffle(rivers)
        rivers = rivers[: self.num_crossings]  # sample random rivers
        rivers_v = sorted(pos for direction, pos in rivers if direction is v)
        rivers_h = sorted(pos for direction, pos in rivers if direction is h)
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1])
                )
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1])
                )
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
