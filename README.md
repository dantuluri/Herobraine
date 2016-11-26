# Herobraine
![Herobraine](https://hydra-media.cursecdn.com/minecraft.gamepedia.com/thumb/b/b4/Herobrine.png/150px-Herobrine.png?version=70c3a6ffaf71ed5e40e000dece271c56)
This repository is the cooresponding codebase for an experiment which 
tests the state of the art in deep reinforcement learning against an entirely
open world in environment: Minecraft.

The primary question of this experiment is as follows. Can we develop AI using
only the reinforcement signal of "experience" and "novelty", when run long
over a long enough duration? To the authors knowledge, there is no such
similar long term application of algorithms like DDPG and A3C to long
term reinforcement learning--and well, this is a fun project anyway.

## The Project Design

The experiment will be broken into a series of phases in which a new
agent (so to speak) is introduced to one contiguous narrative. Technically
we will continuously maintain a Minecraft server (version 1.8) and let
these agents interact with the server and eachother as they learn according
to the regime we provide. There really isn't a goal here, we just want to see 
what the hell happens when you actually stress tess these agents. We'd be
really happy to see the community contribute other agents to the narrative,
but for now we're attempting to stress test the standard set of algorithms.

