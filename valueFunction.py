from ast import List, Tuple
import numpy as np
from beliefSpace import PROBLEM
from followerValueFunction import FollowerValueFunction
from leaderValueFunction import LeaderValueFunction
from problem import PROBLEM
from decisionRule import DecisionRule
from jointAlphaVector import JointAlphaVector
from utilities import *
from alphaVector import AlphaVector
from betaVector import BetaVector
import sys
PROBLEM = PROBLEM.get_instance()
class ValueFunction:
    def __init__(self,belief_space,gametype,sota=False):
        #change this to initialize belief space here
        self.horizon = belief_space.horizon
        self.gametype = gametype
        self.belief_space = belief_space
        self.sota=sota
        self.folower_value_fn = FollowerValueFunction(self.belief_space,gametype,sota)
        self.leader_value_fn = LeaderValueFunction(self.belief_space,gametype,sota)

       
#===== private methods ===============================================================================================


    def get_initial_value(self):
        """function that returns the leaders and followers value at the initial belief (timestep 0)"""
        return self.leader_value_fn.get_initial_value(), self.folower_value_fn.get_initial_value()
    

#===== public methods ===============================================================================================

    def get_alpha_pairs(self,timestep,belief_id):
        return JointAlphaVector(self.leader_value_fn.get_vector_at_belief(belief_id,timestep),self.folower_value_fn.get_vector_at_belief(belief_id,timestep),belief_id)
    
    
    def solve(self,belief_id,timestep) -> tuple[AlphaVector,AlphaVector]:
        """solve function that solves a subgame at each belief point in the tree by creating beta vectors and using a Mixed integer linear program to solve for the optimal decision rule"""
        
        # construct beta vectors
        beta_vector = BetaVector(self.leader_value_fn.construct_beta(belief_id,timestep),self.folower_value_fn.construct_beta(belief_id,timestep))

        # solve subgame using mixed integer linear program (MILP)
        max_plane_leader_value, max_plane_follower_value ,decision_rule = MILP_solve(beta_vector, self.belief_space.get_belief(belief_id), self.sota, self.gametype)

        # reconstruct alpha vectors  using optimal decision rule and beta-vector, then add them to the value function of respecive agents 
        leader_alpha,follower_alpha = reconstruct_alpha_vectors(belief_id,beta_vector,decision_rule)

        return leader_alpha,follower_alpha
    
    def backup(self,belief_id,timestep):
        "backup function for each belief id at a certain timestep"
        leader_alpha,follower_alpha  = self.solve(belief_id,timestep)
        self.leader_value_fn.add_alpha_vector(leader_alpha,timestep)
        self.folower_value_fn.add_alpha_vector(follower_alpha,timestep)

 
