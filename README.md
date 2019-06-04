### How to use it

Copy this repo in `gym/gym/envs/`.

You can register an environment either in the `__init__.py` file in `gym/gym/env`,
or in your code before initializing the environment (before `gym.make`).

In the first case, add (example for the LQR)
```
register(
    id='Lqr-v0',
    entry_point='gym.envs.gym_toy:LqrEnv',
    max_episode_steps=150,
    kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : 20.},
)
```

In the second case,
```
    gym.envs.register(
         id='Lqr-v0',
         entry_point='gym.envs.gym_toy:LqrEnv',
         max_episode_steps=150,
         kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : 20.},
    )
```

For `SparseCar` and `SparseNavi` you will get the following warning because the
initial position is fixed: `UserWarning: WARN: Could not seed environment`.


### Brief description of the environments
* `lqr.py`        : linear-quadratic regulator.
* `sparse_car.py` : car moving on a 1D plane with sparse reward.
* `sparse_navi.py`: agent navigating on a 2D environment, with linear dynamics and sparse reward.
