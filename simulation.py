"""

"""
#
# TODO: Calculate the feature expectations of a given policy -- we'll need todo this for the
# expert's features and some random policy

from collections import Counter
import numpy as np

class Simulation:

    def __init__(self, agents):
        self.agents = self._agent_init(agents=agents)
        self.random_policies = None

    def _agent_init(self, agents):
        """

        Args:
            agents:

        Returns:

        """

        if not type(agents) == list:
            agents = [agents]

        # First collecting a count of agent types
        count_dict = Counter([agent.type for agent in agents])

        # Subsetting the count_dict to a dict that contains duplicate Agent.type
        subset = {k: v for k, v in count_dict.items() if v > 1}

        if len(subset) > 0:
            raise AttributeError(f'You may only have one type of each agent. You have: \n {subset}')

        # If there are no duplicate types, return a dict that will help to reference
        return {agent.type: agent for agent in agents}

    def μ_estimate(self, trajectories, gamma):
        """"""

        if not type(trajectories) == list:
            trajectories = [trajectories]

        if not type(gamma) == float:
            raise TypeError('Gamma must be a float.')

        # Looping through the list of trajectories...
        for traj in trajectories:

            # Now looping through each step in the traj
            # This data should be structured like a dictionary:
            # {'a': 0, 'b': 2, 'c': 3}
            # We use the enumerate() functionality, as the i from the counter
            # will ultimately become the exponent for the discount factor, γ
            for i, step in enumerate(traj):

                # We need to ensure that the ϕ vector is in the same order
                # We can do this by looping through each of the
                phi = np.array([step[x] for x in self.agents['expert'].action_list])

                # Now to do



                print(phi)




    def gen_random_trajectory(self, i, n=1):
        """

        Args:
            i (int): the number of time steps to generate in the trajectories
            n (int): the number of trajectories to generate

        Returns:

        """

        if not 'expert' in [k for k, v in self.agents.items()]:
            raise AttributeError('You do not have an expert agent in your simulation.')


        if not type(i) == int:
            raise TypeError('i must be an int.')

        if not type(n) == int:
            raise TypeError('n must be an int.')

        # Pulling out the action list from the expert agent
        action_list = self.agents['expert'].action_list

        if not n == 1:

            traj_list = []

            for traj in range(n):

                traj_list.append([np.random.choice(action_list, 1)[0] for x in range(i)])

            self.random_policies = traj_list

        else:

            self.random_policies = [np.random.choice(action_list, 1)[0] for x in range(i)]





