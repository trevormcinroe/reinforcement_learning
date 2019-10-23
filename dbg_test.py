import numpy as np
from agents import Agent
from state import Environment
from simulation import Simulation

# Init the Agent's environment
env = Environment()

# Init the expert agent
# Feed it the expert trajectories
a = Agent(type='expert',
          action_list=['l', 'r'],
          environment=env,
          trajectories=[['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']])

# Build said expert trajectories
a.build_trajectories()

# Build the Agent's initial state distribution
a.build_D()

# Init a standalone environment for the state itself
simul_env = Environment()

# Init the simulation
sim = Simulation(agents=a, environment=simul_env, alpha=1)

# This method will initalize a matrix of state-action pairs and their values (currently set to init all
# at 0). This  will build a matrix that represents all of the states that the expert agent has visited
sim.reset_q(trajectories=sim.agents['expert'].state_trajectories)

w = np.array([-0.1, 0.9])
w = w / np.sum(np.abs(w))
sim.q_learning(break_condition=.00001,
               num_steps=10,
               epsilon=.1,
               w=w,
               gamma=0.99)