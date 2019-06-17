import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----

Very simple 2d plane.
The state is the agent position (x,y) and velocity (xd, yd), the bonus location (bx, by), goal location and an indicator if the bonus was picked up.
The action is the acceleration in x/y direction.
The location of the bonus is randomly chosen at the beginning of the episode among
three possible positions: either behind the initial position of the agent, behind the goal or in between.
The initial position is fixed, as well as the goal position.
The agent gets the bonus if it is very close to it with almost 0 velocity. The bonus state gets set to +1 from initial -1.
The episode ends when the agent is very close to the goal with almost 0 velocity or after the defined max_episode_steps.
Reward only for getting to the goal. Doubled reward if he got the bonus prior.


---- HOW TO USE ----

* Place this file in gym/gym/envs/classic_control
* Add to __init__.py (located in the same folder)

    from gym.envs.classic_control.toy import ToyChainSparseEnv2D

* Register the environment in your script

    gym.envs.register(
         id='ToyChainSparse-v0',
         entry_point='gym.envs.classic_control:ToyChainSparseEnv2D',
         max_episode_steps=1000,
    )
'''

class ToyChainSparseEnv2D(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.dt = 0.1
        self.tol = 0.01
        self.goal_state = [0.6, 0]
        self.bonus_states = [[-0.5, 0.1],
                             [0.25, 0],
                             [0.8, -0.1]]
        self._seed()
        self._step = 0
        self.step_size = 1.0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        self._step += 1
        u = np.clip(u, self.action_space.low, self.action_space.high)
        x = self.state[0] # player pos x
        y = self.state[1] # player pos y
        xd = self.state[2] # player velocity x
        yd = self.state[3] # player velocity y
        bx = self.state[4] # bonus pos x
        by = self.state[5] # bonus pos y
        gb = self.state[6] # got bonus?
        xd_n = xd + u[0]*self.dt
        x_n = x + xd#*self.dt # already u*dt
        yd_n = yd + u[1]*self.dt
        y_n = y + yd#*self.dt # already u*dt

        done = False
        rwd = 0.
        dist_goal = np.abs(x - self.goal_state[0]) + np.abs(y - self.goal_state[1])
        dist_bonus = np.abs(x - bx) + np.abs(y - by)
        if dist_bonus < self.tol and np.abs(xd) < self.tol and np.abs(yd) < self.tol:
            gb = 1 # set bonus collected #b = self.goal_state # place bonus where the goal is
        if dist_goal < self.tol and np.abs(xd) < self.tol and np.abs(yd) < self.tol:
            rwd = 1 # overwrite bonus reward with goal reward (because you already collected it)
            done = True
            if gb > 0:
                rwd *= 2
        if np.abs(x_n) > 1: # collision with world border
            rwd -= 0.1 # bad boy
            xd_n = 0   # set velocity 0
            yd_n = 0   # set velocity 0
        if np.abs(y_n) > 1: # collision with imaginary street border obstacles
            rwd -= 0.1 # bad boy
            xd_n = 0   # set velocity 0
            yd_n = 0   # set velocity 0

        # After the max step size we give the end bonus for the distance to goal. Double if we got the bonus.
        if self._step >= self.spec.max_episode_steps and not done:
            mult = 1.0
            if gb > 0:
                mult = 2.0
            rwd += (dist_goal / self.start_dist) * mult
            done = True

        self.state[0] = x_n#np.clip(x_n, self.observation_space.low, self.observation_space.high)
        self.state[1] = xd_n
        self.state[2] = y_n
        self.state[3] = yd_n
        self.state[6] = gb
        self.state[9] = self.step_size * self._step
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        return self.state, rwd, done, {}

    def reset(self):
        self._step = 0
        b = np.random.randint(0,3)
        self.state = np.array([0, # player pos x
                               0, # player pos y
                               0, # player velocity x
                               0, # player velocity y
                               self.bonus_states[b][0], # bonus pos x
                               self.bonus_states[b][1], # bonus pos y
                               -1, # got bonus?
                               self.goal_state[0], # goal pos x
                               self.goal_state[1], # goal pos x
                               0 # step
                               ])
        self.step_size = 1.0 / self.spec.max_episode_steps
        self.last_u = None
        self.start_dist = np.abs(self.state[0] - self.goal_state[0]) + np.abs(self.state[1] - self.goal_state[1])
        return self._get_obs()

    def _get_obs(self):
        return self.state