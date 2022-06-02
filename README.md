# What is this?
This is an implementation of my masters thesis. It relies on the `tensortrade` for setting up financial RL environments and `Rlib` for RL agent implementations. My masters thesis has the goal of combining multiple trading RL algorithms, each trainined on different data types, to create a more sofisticated and greater performing algorithm. To achieve this goal, I will be creating trading strategies. Each trading strategy will produce an RL algorithm. Strategies may encorporate previous strategies. 

# Repository Structure
- **.vscode** contains opinionated vscode settings for debugging and formatting.
- **experiments** include jupyter notebooks created for investigation purposes of the masters. (This could be removed)
- **strategies** strategies implemented to train DRL agents on specific data.
- **models** saved models from tested strategies (these may be excluded from git due to size).
- **logs** training logs for strategies.
- **rayExtensions** classes which extend the behaviour of the ray framework. 
- **tensortrade** extensios made to plug into the tensortrade framework. 


# Install Instructions
1. Install Anaconda and create a new environment for this projects dependancies
2. Attemp to install from the environment.yml for conda. If it works great, if it doesn't, follow the next steps.
3. Install Ray RLlib from the official installation instructions
4. Install tensortrade

# Applic silicon (m1), Install Instructions
1. Install Anaconda and create a new environment for this projects dependancies
2. Attemp to install from the environment.yml for conda. If it works great, if it doesn't, 
3. Install Ray Rlib, follow the apple silicon instructions from the official website
4. Install tensortrade for apple silicon
5. Download the tensortrade repository, comment out tensotrade from the requirements.txt and install all depedencies.
6. You're good to go.


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