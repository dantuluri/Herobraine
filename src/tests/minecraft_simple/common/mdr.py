"""
mdr.py - MultiDiscrete random process.
Author: William GUss
"""
import numpy.random as nr
import numpy as np

class MultiDiscreteRandom:
    """A Multidiscrete random uniform process.
    	This essentially just packages n-random levers."""
    def __init__(self, action_dimension, exploration_rate=1e-6, mu=0.5, rng=0.5, steady=0.5):
        self.action_dimension = action_dimension
        self.exploration_rate = exploration_rate
        self.mu = mu
        self.rng = rng
        self.steady = 0.5
        self.time = 0
        self.state = None

    def reset(self):
    	self.state = None

    def noise(self, action):
        self.time += 1
        should_random = nr.random(1) > float(self.time)*self.exploration_rate
        if should_random:
            if self.state is None or nr.random(1) > self.steady:
                self.state = np.round(2*self.rng * (nr.rand(self.action_dimension) - 0.5) + self.mu)
            return self.state
        else:
            return action

if __name__ == '__main__':
    mdr = MultiDiscreteRandom(2, rng=10000)
    states = []
    for i in range(1000):
        states.append(mdr.noise())
    states = np.array(states)
    print("MDR Test:")
    print(np.max(states, axis=0))
    print(np.min(states, axis=0))
    print(np.mean(states, axis=0))