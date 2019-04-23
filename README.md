# Quixo

Experimentation with Deep Reinforcement learning on the example of teaching how to play [Quixo](https://boardgamegeek.com/thread/451817/quixo-detailed-review).

Two approaches will be tested:
* dqn (Deep Q Network)
* ddqn (Double Deep Q Network).

First they will learn how to play by playing against rand_bot (agent playing random but valid moves). 
Then teaching will continue by playing against their older versions.

After the training is completed, the effectiveness of the results will be checked through the tournament against minimax_bot 
(bot using [minimax](https://en.wikipedia.org/wiki/Minimax) with some improvements like 
[alpha-beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
or [iterative deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search)).
