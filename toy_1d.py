import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

'''
---- DESCRIPTION ----

Very simple chainwalk with linear dynamics.
The state is the agent position (x) and velocity (xd), and the bonus location (b).
The action is the acceleration.
The location of the bonus is randomly chosen at the beginning of the episode among
three possible positions: either behind the initial position of the agent, behind the goal or in between.
The initial position is fixed, as well as the goal position.
The agent gets the bonus if it is very close to it with almost 0 velocity. The bonus state gets set to +1 from initial -1.
The episode ends when the agent is very close to the goal with almost 0 velocity or after the defined max_episode_steps.
Reward only for getting to the goal. Doubled reward if he got the bonus prior.


---- HOW TO USE ----

* Place this file in gym/gym/envs/classic_control
* Add to __init__.py (located in the same folder)

    from gym.envs.classic_control.toy import ToyChainSparseEnv

* Register the environment in your script

    gym.envs.register(
         id='ToyChainSparse-v0',
         entry_point='gym.envs.classic_control:ToyChainSparseEnv',
         max_episode_steps=1000,
    )
'''

class ToyChainSparseEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.dt = 0.1
        self.tol = 0.01
        self.goal_state = 0.6
        self.bonus_states = [-0.5, 0.25, 0.8]
        self._seed()
        self._step = 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        self._step += 1
        u = np.clip(u, self.action_space.low, self.action_space.high)
        x = self.state[0] # player pos
        xd = self.state[1] # player velocity
        b = self.state[2] # bonus pos
        gb = self.state[4] # got bonus?
        xd_n = xd + u*self.dt
        x_n = x + xd#*self.dt # already u*dt

        done = False
        rwd = 0.
        dist_goal = np.abs(x - self.goal_state)
        dist_bonus = np.abs(x - b)
        if dist_bonus < self.tol and np.abs(xd) < self.tol:
            gb = 1 # set bonus collected #b = self.goal_state # place bonus where the goal is
        if dist_goal < self.tol and np.abs(xd) < self.tol:
            rwd = 1 # overwrite bonus reward with goal reward (because you already collected it)
            done = True
            if gb > 0:
                rwd *= 2
        if np.abs(x_n) > 1: # collision with world border
            rwd -= 0.1 # bad boy
            xd_n = 0   # set velocity 0

        # After the max step size we give the end bonus for the distance to goal. Double if we got the bonus.
        if self._step >= self.spec.max_episode_steps and not done:
            mult = 1.0
            if gb > 0:
                mult = 2.0
            rwd += (dist_goal / self.start_dist) * mult

        self.state[0] = x_n#np.clip(x_n, self.observation_space.low, self.observation_space.high)
        self.state[1] = xd_n
        self.state[4] = gb
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        return self._get_obs(), rwd, done, {}

    def reset(self):
        self.state = np.array([0,0,0,self.goal_state,-1])
        self.state[2] = self.bonus_states[np.random.randint(0,2)]
        self.last_u = None
        self._step = 0
        self.start_dist = np.abs(self.state[0] - self.goal_state)
        return self._get_obs()

    def _get_obs(self):
        return self.state