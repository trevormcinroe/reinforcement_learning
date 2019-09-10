"""

"""
#
# TODO: Calculate the feature expectations of a given policy -- we'll need todo this for the
# expert's features and some random policy

from collections import Counter

class Simulation:

    def __init__(self, agents):
        self.agents = self._agent_init(agents=agents)


    def _agent_init(self, agents):
        """"""

        # First collecting a count of agent types
        count_dict = Counter([agent.type for agent in agents])

        # Subsetting the count_dict to a dict that contains duplicate Agent.type
        subset = {k: v for k, v in count_dict.items() if v > 1}

        if len(subset) > 0:
            raise AttributeError(f'You may only have one type of each agent. You have: \n {subset}')

        # If there are no duplicate types, return a dict that will help to reference
        return {agent.type: agent for agent in agents}

    def μ_estimate(self):
        """"""

        if not 'expert' in [k for k, v in self.agents.items()]:
            raise AttributeError('You do not have an expert agent in your simulation.')

        # Pulling out the expert Agent
        expert = self.agents['expert']






