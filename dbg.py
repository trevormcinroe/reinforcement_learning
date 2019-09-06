from agents import Agent
from simulation import Simulation
from state import  Environment

env = Environment()

a = Agent(type='expert',
          action_list=['a', 'b', 'c'],
          environment=env,
          trajectories=[['a', 'b', 'c', 'b'],
                        ['a', 'b', 'c', 'a'],
                        ['c', 'b', 'c', 'a']])

a._build_trajectories()

a._build_D()