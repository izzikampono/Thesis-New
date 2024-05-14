from __future__ import annotations
import numpy as np
from problem import PROBLEM
PROBLEM = PROBLEM.get_instance()

class AlphaVector:
    """ Class representing AlphaVectors : state-value functions at a given belief state of the game
        Attributes:
        belief_id = id of the belief state in which the alpha vector was built on
        decision_rule = the decision rule associated with the construction of this alpha-vector
        vector = state-value function itself represented as a numpy vector , i.e. V(b) = [V_b(x_0),...,V_b(x_n)]
    """
    
    def __init__(self,decision_rule,vector,belief_id):
        self.belief_id = belief_id
        self.decision_rule = decision_rule
        self.vector = vector

    def get_value(self,belief):
        """returns the value of an alpha-vector at a certain belief by conducting a vector product multiplication"""
        return np.dot(belief.value,self.vector)

    def print(self):
        print(self.vectors)



