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
    def __init__(self):
        """"""

        # self.action_list = action_list
        self.current_state = None

    def _reset(self, action_list):
        """A function that will reset the State to initial conditions"""

        if not type(action_list) == list:
            raise TypeError('Given action_list must be in the form of a list')

        self.current_state = {
            x: 0 for x in action_list
        }


    def _update_state(self, action, ret=False):
        """A function that updates our State from s --> s' """

        if not action in [k for k, v in self.current_state.items()]:
            raise AttributeError(f'Given action -- {action} -- not available in the current state.')

        self.current_state[action] = self.current_state[action] + 1

        if ret:
            return self.current_state



