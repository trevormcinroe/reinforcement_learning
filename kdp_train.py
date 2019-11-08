import numpy as np
import pandas as pd
import os
from agents import Agent
from state import Environment
from simulation import Simulation

#############################
#=== Expert Trajectories ===#
#############################
from expert_aspects import e_action_list, e_trajs, e_action_map

# Creating an  environment for the expert trajectories
e_env = Environment(attribute_mapping=e_action_map)

# Init the expert agent
# Feed it the expert trajectories
a = Agent(type='expert',
          action_list=e_action_list,
          environment=e_env,
          trajectories=e_trajs)


# a.environment._reset(action_list=a.action_list, attribute_based=True)
#
#
# a.environment._update_state(action='1', attribute_based=True)
#
# print(e_action_map['1'])
# print(a.environment.current_state)

# Build said expert trajectories
a.build_trajectories(attribute_based=True)

# # Build the Agent's initial state distribution
a.build_D()


print(a.state_trajectories)