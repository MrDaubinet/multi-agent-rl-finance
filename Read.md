# What is this?
This is an implementation of my masters thesis. It relies on the `tensortrade` for setting up financial RL environments and `stable-baselines-3` for RL agent implementations. My masters thesis has the goal of combining multiple trading RL algorithms, each trainined on different data types, to create a more sofisticated and greater performing algorithm. To achieve this goal, I will be creating trading strategies. Each trading strategy will produce an RL algorithm. Strategies may encorporate previous strategies. 

# Repository Structure
- **.devcontainer** contains vscode settings for running this project as a docker contain in vscode.
- **.vscode** contains opinionated vscode settings for debugging and formatting.
- **Experiments** include jupyter notebooks created for investigation purposes of the masters.
- **models** saved models from tested strategies (these may be excluded from git due to size).
- **logs** training logs for strategies.
- **tensortrade** extensios I make to plug into the tensortrade framework. 


# Install Instructions
## Docker
I am programming on an apple m1 arm based machine. Due to this, I had to build a docker image `tf-devel-cpu-arm64v8-jupyter` for tensorflow that is based off of Arm. I did this buy downloading the tensorflow repository and running docker build for the the `arm64v8` dockerfile. This could probably be automated with a bash script, but your cpu architecture may change. Update the base docker image to a tensorflow base image supporte by your pc.

## Environment paths
I had to set my LD_PRELOAD path for torch with the following:

```
  export LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/torch/lib/libgomp-d22c30c5.so.1
```

## Install TA-LIB
Download tar file and run `make` then `make install`.