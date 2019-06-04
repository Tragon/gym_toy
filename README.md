### How to use it

Copy this repo in `gym/gym/envs/`.

You can register an environment either in the `__init__.py` file in `gym/gym/env`,
or in your code before initializing the environment (before `gym.make`).

Here is an example
```
    gym.envs.register(
         id='Lqr-v0',
         entry_point='gym.envs.gym_toy:LqrEnv',
         max_episode_steps=150,
         kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : 20.},
    )
```


### Brief description of the environments
* `lqr.py`      : linear-quadratic regulator.
* `sparse1d.py` : gent moving on a 1D plane with sparse reward.
* `sparse2d.py` : like LQR, but with sparse reward.
