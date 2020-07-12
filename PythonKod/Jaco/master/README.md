# gym-metak

An OpenAI gym environment that can communicate with pybullet.

*"aj sad lipo faco digni auto i vrati ga nazad di je bia"* - metak metak, 2008.



# Installation

This package requires quite a few dependencies. To install, do the following, in the exact listed order.

1. Activate a virtual environment (just to be on the safe side)
2. Install the [bullet3](https://github.com/bulletphysics/bullet3) package (caution: you are required to build this after using git-clone)
3. Install the [baselines](https://github.com/openai/baselines/) package (caution: follow the github instructions, some extras need to be installed)
4. Install the [liegroups](https://github.com/utiasSTARS/liegroups) package
5. Install the [pyb-manipulator](https://github.com/utiasSTARS/pyb-manipulator) package
6. Install gym-metak by cloning and running ```pip install -e .```
7. Once installed, go to ```./baselines/baselines/run``` and add the line ```import gym_metak``` to the beginning of the file

# Usage

The main purpose of this environment is training RL models. This is done using the ```baselines``` package. Before using, study the documentation on the package provided by OpenAI. It can be found on the project github page.
To train a model, run the ```python -m baselines.run --env='metak-v0' --alg=ppo2 --num_timesteps=1e6``` command, freely choosing the algorithm and number of timesteps. Using additional arguments, the user can configure the network type, learning parameters, batch size etc. 
A list of arguments and their respective values can be found in the OpenAI documentation.

#TRENIRANJE:
1. python3 -m baselines.run --alg=ppo2 --env='metak-v0' --num_timesteps=2e7 --save_path=~/models/newobsprogdisspe_2e7_ppo2 --log_path=~/logs/newobsprogdisspe_2e7_ppo2  --nminibatches=32 --noptepochs=10 --num_env=128 --value_network=copy --log_interval=1
2. python3 -m baselines.run --alg=ppo2 --env='metak-v0' --num_timesteps=4e7 --save_path=~/models/natrecuobsprogdisspe_4e7_ppo2 --log_path=~/logs/natrecuobsprogdisspe_4e7_ppo2  --nminibatches=32 --noptepochs=10 --num_env=128 --value_network=copy --log_interval=1
3. python3 -m baselines.run --alg=ppo2 --env='metak-v0' --num_timesteps=4e7 --save_path=~/models/svilinkoviobsprogdisspe_4e7_ppo2 --log_path=~/logs/svilinkoviobsprogdisspe_4e7_ppo2  --nminibatches=32 --noptepochs=10 --num_env=128 --value_network=copy --log_interval=1


#POKRETANJE:
1. python3 -m baselines.run --alg=ppo2 --env='metak-v0' --num_timesteps=0 --load_path=./../models/newobsprogdisspe_2e7_ppo2 --play
2. python3 -m baselines.run --alg=ppo2 --env='metak-v0' --num_timesteps=0 --load_path=./../models/natrecuobsprogdisspe_4e7_ppo2 --play
3. python3 -m baselines.run --alg=ppo2 --env='metak-v0' --num_timesteps=0 --load_path=./../models/svilinkoviobsprogdisspe_4e7_ppo2 --play
