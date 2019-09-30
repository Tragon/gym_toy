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
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.dt = 0.1
        self.tol = 0.01
        self.goal_state = 0.6
        self.bonus_states = [-0.5, 0.25, 0.8]
        self._seed()
        self._step = 0
        self.step_size = 1.0
        self.speed_penalty_factor = 0#.01
        self.wall_penalty_factor = 0#.01
        self.goal_reward = 1.0
        self.bonus_reward = 1.0
        self.partial_reward_factor = 1.0
        max_rwd = self.bonus_reward + self.goal_reward
        self.reward_range = (-float(max_rwd), float(max_rwd))#spaces.Box(low=-max_rwd, high=max_rwd, shape=(1,), dtype=np.float32)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        simulate = False
        if isinstance(u, list) and len(u) > 1:
            simulate = u[1]
            u = u[0]
        u = np.clip(u, self.action_space.low, self.action_space.high) * self.dt
        x = self.state[0] # player pos
        xd = self.state[1] # player velocity
        b = self.state[2] # bonus pos
        gb = self.state[3] # got bonus?
        xd_n = xd + u[0]
        x_n = x + xd*self.dt # already u*dt

        done = False
        rwd = 0.0
        dist_goal = np.abs(x - self.goal_state)
        dist_bonus = np.abs(x - b)
        if dist_bonus < self.tol and np.abs(xd) < self.tol:
            gb = 1.0 # set bonus collected #b = self.goal_state # place bonus where the goal is
        if dist_goal < self.tol and np.abs(xd) < self.tol:
            rwd = self.goal_reward # overwrite bonus reward with goal reward (because you already collected it)
            done = True
            if gb > 0:
                rwd += self.bonus_reward
        if np.abs(x_n) > 1: # collision with world border
            rwd -= 1 * self.wall_penalty_factor # bad boy
            xd_n = 0.0   # set velocity 0

        # After the max step size we give the end bonus for the distance to goal. Double if we got the bonus.
        if self._step >= self.spec.max_episode_steps-1 and not done:
            mult = 1.0
            part_rwd = (1.0 - (dist_goal / self.start_dist)) * self.goal_reward * self.partial_reward_factor
            if gb > 0: #and part_rwd > 0
                #mult = 2.0
                rwd += self.bonus_reward
            rwd += part_rwd * mult
            done = True

        rwd -= np.abs(xd_n) * self.speed_penalty_factor # penalize speed -> go slower for exploration

        new_state = self.state
        new_state[0] = x_n#np.clip(x_n, self.observation_space.low, self.observation_space.high)
        new_state[1] = xd_n
        new_state[3] = gb
        #self.state[4] = self.step_size * self._step
        new_state = np.clip(new_state, self.observation_space.low, self.observation_space.high)
        if simulate:
            self._step += 1
            self.state = new_state
        return new_state, rwd, done, {}

    def reset(self):
        self._step = 0
        bonus_pos = self.bonus_states[np.random.randint(0,3)]
        self.state = np.array([0.0, # pos
                               0.0, # velocity
                               bonus_pos, # bonus pos
                               -1.0 # got bonnus?
                               #,self.goal_state
                               ])
        self.step_size = 1.0 / self.spec.max_episode_steps
        self.last_u = None
        self.start_dist = np.abs(self.state[0] - self.goal_state)
        return self._get_obs()

    def _get_obs(self):
        return self.state