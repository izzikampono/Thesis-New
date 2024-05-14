from ast import Dict
import numpy as np
from alphaVector import AlphaVector
from problem import PROBLEM
PROBLEM = PROBLEM.get_instance()
class JointAlphaVector:
    """ Holder class that keeps a pair of alphaVectors, i.e the leader's alpha vector and the followers alpha vector at a certain point of the game"""
    
    def __init__(self,leader_alpha,follower_alpha,belief_id):
        self.individual_vectors = [leader_alpha,follower_alpha]
        self.belief_id = belief_id

    def get_leader_vector(self) -> AlphaVector:
        return self.individual_vectors[PROBLEM.LEADER]

    def get_follower_vector(self)-> AlphaVector:
        return self.individual_vectors[PROBLEM.FOLLOWER]

    def get_value(self,belief):
        """returns the value of an alpha-vector at a certain belief by conducting a vector product multiplication"""
        return np.dot(belief.value,self.individual_vectors[0].vector),np.dot(belief.value,self.individual_vectors[1].vector)

    # def get_vectors(self):
    #     return self.individual_vectors[0].vector , self.individual_vectors[1].vector



        



