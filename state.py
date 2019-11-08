"""

"""

import numpy as np
import pandas as pd

class Environment:
    """
    That State class will contain all of the information pertaining to the State of the
    RL simulation

    Attributes:
        action_list (list):
        current_state (dict):

    """

    def __init__(self, attribute_mapping):
        """"""

        # self.action_list = action_list
        self.current_state = None
        self.attribute_mapping = attribute_mapping

    def _reset(self, action_list, attribute_based=False):
        """A function that will reset the State to initial conditions"""

        if not type(action_list) == list:
            raise TypeError('Given action_list must be in the form of a list')

        if not attribute_based:

            self.current_state = {
                x: 0 for x in action_list
            }

        else:

            # Pulling out the attributes
            self.current_state = {
                str(x): 0 for x in range(len(self.attribute_mapping[[k for k, v in self.attribute_mapping.items()][0]]))
            }


    def _update_state(self, action, ret=False, attribute_based=False):
        """A function that updates our State from s --> s' """

        if not action in [k for k, v in self.attribute_mapping.items()]:
            raise AttributeError(f'Given action -- {action} -- not available in the current state.')

        if not attribute_based:

            self.current_state[action] = self.current_state[action] + 1

            if ret:
                return self.current_state

        else:

            # Need to find the corresponding attribute vector for the given action
            attribute_vec = self.attribute_mapping[action]

            for i in range(len(attribute_vec)):

                self.current_state[str(i)] += attribute_vec[i]

            if ret:
                return self.current_state

