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

# TODO:
* update from the BSH action scheme to a BHSS (short) scheme which selects specified blocks of shares.
  * blocks should be equvalent to maximium cash in shares normalised between 1 - 10.
    * E.g, 10 dollars = 10 shares, therefore trading quantities = [1, 2, 3, 4, 5, 6, 7, 8 , 9, 10] for buy, sell and short.
* update reward scheme from simple profit to something which includes risk, position-time and other useful evaluators identified in my research.
* update network architecture to use a cnn
* update network architecture to use an LSTM
* implement a strategy on stock data
  * Setup preprocessing:
    * generate technical indicators
    * identify corelation & remove duplicates
* implement a strategy on fundamental data
* implement a strategy which utilises multi-agent rl
* implement a strategy which utilises hierarchical rl for quantity selection and stock action
  * implement a new strategy as an action selection strategy which specifies an amount of stock to buy / sell / short

**future additions**
* Add a stock picking rl agent who's trained to select the best stock at a specified period
  * Add this rl agent as a new level to the hierarchical rl agent. 