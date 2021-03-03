# Slither.io Actor-Critic Agent

* [**YouTube demo of the agent in action**](https://www.youtube.com/watch?v=fhAlw9w-MNk)

Turn off any settings that would start your screensaver or turn off your display while running this code. The script needs the display to be on in order to take continuous screenshots of the game state.

As expected, it's highly difficult for an RL algorithm to learn when pinned against live human players, for they have highly stochastic behavior. For this reason, the agent learns a very defensive policy, ie. It goes to the less-populated side of the map to avoid interaction because it is pessimistic in terms of its reward function when another player is present onscreen. It achieves a maximum score of approximately 600, decent considering its learning from pixels alongside online players. 
