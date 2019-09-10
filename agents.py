""""""

import numpy as np
#TODO: Ensure that the trajectories arg is a list of lists

#TODO: finish _build_D() method that ensures exploration. Is this needed??

class Agent:

    """
    Attributes:
        type:
        action_list
        environment
        trajectories
    """


    def __init__(self, type, action_list, environment, trajectories=None):
        self.type = type
        self.action_list = action_list
        self.environment = environment
        self.trajectories = trajectories
        self.state_trajectories = None
        self.D = None
        self.T = None  # Might not even need this with certain RL algos

    def build_trajectories(self):
        """
        Using a list of given action lists from an "expert", this function will build the Trajectory of the States
        after each action in the list has been taken.

        For convenience sake, the first element of the list is an initial state, with every action count set to 0
        """

        # Natural break for when the trajectories == None
        if self.trajectories is None:
            return None

        # Checking whether or not trajectories have been supplied
        if len(self.trajectories) == 0:
            raise ValueError('No trajectories have been supplied.')

        # Need to first check whether or not the current agent is an Expert
        if not self.type == 'expert':
            raise AttributeError(f'Your agent must be an expert type to do this. This agent is: {self.type}')

        # A catch in the beginning to ensure that the user is wanting to reset the environment
        answer = input('Building the State Trajectories will cause the attached environment to '
                       'be reset. Are you sure? (y/n)')

        if not answer == 'y':
            print('Aborting State Trajectory creation.')
            return None


        # Init a list that will hold all
        all_traj_holder = []

        # Looping through each given trajectory within the trajectories list
        for trajectory in self.trajectories:

            # If the user does wish to do so, hit the reset button on the environment
            self.environment._reset(action_list=self.action_list)

            # Init an empty list
            traj_l = []

            # Appending in our S_0
            # There  is excessive use of the .copy() method here. This is a silly Python thing

            # I have commented out the S0 as being  a vector of 0s.
            # Seems unnecessary to have and complicates the Mu calc? 9.10.2019
            # traj_l.append(self.environment.current_state.copy())

            # Looping through each state and recording it
            for action in trajectory:

                traj_l.append(self.environment._update_state(action=action, ret=True).copy())

            # Appending each trajectory to the outer holder
            all_traj_holder.append(traj_l)

        # Setting the trajectories for the Agent
        self.state_trajectories = all_traj_holder

        # Resetting the environment back to its initial state
        self.environment._reset(action_list=self.action_list)

    def build_D(self, ensure_exploration=False, ε=None):
        """"""

        if self.state_trajectories == None:
            raise AttributeError('Please first build the state trajectories.')

        # If the user has decided to allow exploration to non-expert-visited S_0...
        if ensure_exploration:

            # Need some checks here to make sure that the user has entered ε
            if ε is None:
                raise AttributeError('ε has not been specified.')

            if 1 < ε < 0:
                raise ValueError('ε ∈ [0,1]')

            # Init a dictionary that will be filled
            base_dict = {x: 0 for x in self.action_list}

            # Looping through each empirical trajectory and adding one to the counter for each
            # initial action
            for traj in self.trajectories:
                base_dict[traj[0]] = base_dict[traj[0]] + 1

            # Looping through now and normalizing the counts
            for k, v in base_dict.items():

                base_dict[k] = round(v / len(self.trajectories), 3)

            # As we can see, actions that have never been taken are now action: 0.0%
            # Let's raise these up to the given percentage and uniformly reduce the
            # other actions by the sum of this amount


        # If the user decides to not allow for initial state exploration, we simply use the
        else:

            # Init a dictionary that will be filled
            base_dict = {x: 0 for x in self.action_list}

            # Looping through each empirical trajectory and adding one to the counter for each
            # initial action
            for traj in self.trajectories:

                base_dict[traj[0]] = base_dict[traj[0]] + 1

            # Looping through now and normalizing the counts
            for k, v in base_dict.items():

                base_dict[k] = round(v / len(self.trajectories), 3)

            # And finally, assigning our dictionary of normalized values to the D attribute
            self.D = base_dict


    def build_transition_probabilities(self):
        """"""

        # Initalizing
        pass

