# Phase 1.

The first phase of this experiment involves creating a baseline
agent which will play and learn minecraft over the course of several months.
We then will use the agent only in inference mode and hopefully see how it
fairs in a multiagent environment.

## The Agent: Delores
In Phase 1, our agent, Delores, will implement the [DDPG algorithm](https://arxiv.org/pdf/1509.02971.pdf).
We plan on making the network architecture very large, think VGG 19 but larger.
Delores will learn purely from pixel input and be rewarded by 
the experience signal and receive a large negative reward for dying.

More on the purposed architecture soon!
