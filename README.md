# Slither.io Actor-Critic Agent

## Usage
Turn off any settings that would start your screensaver or turn off your display while running this code. The script needs the display to be on in order to take continuous screenshots of the game state.

As expected, it's highly difficult for an RL algorithm to learn when pinned against live human players, for they have highly stochastic behavior. For this reason, the agent learns a very defensive policy, ie. It goes to the less-populated side of the map to avoid interaction because it is pessimistic in terms of its reward function when another player is present onscreen. It achieves a maximum score of approximately 600, decent considering its learning from pixels alongside online players. 

## Demo
https://user-images.githubusercontent.com/18452581/116118383-3aaccc00-a68b-11eb-8555-905a8b93b7ef.mp4

https://user-images.githubusercontent.com/18452581/116118903-d0485b80-a68b-11eb-9089-c43440ddd0e3.mp4

[**Watch it on YouTube**](https://www.youtube.com/watch?v=fhAlw9w-MNk)
