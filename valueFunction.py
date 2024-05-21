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
import typing
import sys
PROBLEM = PROBLEM.get_instance()
import gc
gc.enable()

class ValueFunction:
    def __init__(self,horizon,belief_space,gametype,sota=False):
        #change this to initialize belief space here
        self.horizon = horizon
        self.gametype = gametype
        self.belief_space = belief_space
        self.sota=sota
        self.folower_value_fn = FollowerValueFunction(horizon,self.belief_space,gametype,sota)
        self.leader_value_fn = LeaderValueFunction(horizon,self.belief_space,gametype,sota)
        self.initialize_value_function()
        
       
#===== private methods ===============================================================================================


    def get_initial_value(self):
        """function that returns the leaders and followers value at the initial belief (timestep 0)"""
        return self.leader_value_fn.get_initial_value(), self.folower_value_fn.get_initial_value()
    

#===== public methods ===============================================================================================
    def initialize_value_function(self):
        self.vector_sets = {}
        for timestep in range(self.horizon+1):
            self.vector_sets[timestep] = {}
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.vector_sets[timestep][belief_id] = None

 
    def get_max_alpha(self,belief_id,timestep) -> Tuple(JointAlphaVector,int):
        """returns alpha vector object that gives the maximum value for a given belief at a certain timestep, also returns the belief_id of the alphavector"""
        max = -np.inf
        max_alpha = None
        max_value = None
        max_alpha_belief = None
        for belief_id, alpha in self.vector_sets[timestep].items():
            leader_value,follower_value = alpha.get_value(self.belief_space.get_belief(belief_id))
            if leader_value>max :
                max = leader_value
                max_value = (leader_value,follower_value)
                max_alpha = alpha
                max_alpha_belief = belief_id
        return max_alpha,belief_id


        
    def get_alpha_pairs(self,belief_id,timestep):
        return JointAlphaVector(self.leader_value_fn.get_vector_at_belief(belief_id,timestep),self.folower_value_fn.get_vector_at_belief(belief_id,timestep),belief_id)
    
    def get_leader_DR(self,belief_id,timestep):
        return self.leader_value_fn.get_vector_at_belief(belief_id,timestep).decision_rule

    def get_follower_DR(self,belief_id,timestep):
        return self.folower_value_fn.get_vector_at_belief(belief_id,timestep).decision_rule
    
    def construct_beta(self,belief_id,timestep):
        """function to construct beta vectors for a given subgame rooted at a belief_state.
           takes in a belief_state that the subgame is rooted at, and the mappings of future beliefs stemming from belief_state to its corresponding maximum alpha-vectors

            ## pseudo code :
            
            beta = zeros((X,U_joint))
            for x in X:
                for u in U_joint:
                    beta(x,u) = reward(x,u) 
                    for z in Z_joint:
                        b_t+1 = Transition(b_t,u,z)
                        for y in X :
                            beta(x,u) += dynamics(u,z,x,y) * Value_Fn(b_t+1)[y]
        
        """
        #initialize beta
        beta = np.zeros((2,len(PROBLEM.STATES),len(PROBLEM.JOINT_ACTIONS)))
        for agent in range(2):
            for state in PROBLEM.STATES :
                for joint_action in PROBLEM.JOINT_ACTIONS:
                    # beta(x,u) = reward(x,u)
                    beta[agent][state][joint_action] = PROBLEM.REWARDS[self.gametype][agent][joint_action][state]

                    # if statement for beta-vectors rooted at the last gamestage that has no future reward approximation 
                    if timestep+1 >= self.horizon : continue

                    # if statement for blind follower 
                    if (agent == PROBLEM.FOLLOWER and self.gametype=="generalsum" and self.sota==True): continue

                    for joint_observation in PROBLEM.JOINT_OBSERVATIONS:

                        #check for the resulting next belief id from the (U,Z) combination, if flag == True , then the algorithm found an existing belief in the belief space
                        next_belief_id,flag = self.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation,timestep) 
                        
                        # check for observation probability 
                        if next_belief_id is not None:

                            for next_state in PROBLEM.STATES:
                                # beta(x,u)^agent :: Transition(x,u,y,z) * value_fn[T(b,u,z)][y]
                                try:    
                                    beta[agent][state][joint_action] += PROBLEM.TRANSITION_FUNCTION[joint_action][state][next_state] * PROBLEM.OBSERVATION_FUNCTION[joint_action][state][joint_observation] * self.vector_sets[timestep+1][next_belief_id].individual_vectors[agent].vector[next_state]
                            
                                # if next belief id does not exist in the value function, get the maximizing alpha w.r.t that belief state 
                                except:
                                    alpha , alpha_id = self.get_max_alpha(next_belief_id,timestep+1)
                                    # calculate beta using maximized alpha-vector
                                    beta[agent][state][joint_action] += PROBLEM.TRANSITION_FUNCTION[joint_action][state][next_state] * PROBLEM.OBSERVATION_FUNCTION[joint_action][state][joint_observation] * self.vector_sets[timestep+1][alpha_id].individual_vectors[agent].vector[next_state]
                                    # record the connection between newly found belief state and alpha vector and add it to the value function  
                                    self.vector_sets[timestep+1][next_belief_id] = alpha


                                   
        return BetaVector(beta[PROBLEM.LEADER],beta[PROBLEM.FOLLOWER])


    def solve(self,belief_id,timestep) -> tuple[AlphaVector,AlphaVector]:
        """solve function that solves a subgame at each belief point in the tree by creating beta vectors and using a Mixed integer linear program to solve for the optimal decision rule"""
        
        # construct beta vectors
        beta_vector = self.construct_beta(belief_id,timestep)

        # solve subgame using mixed integer linear program (MILP)
        max_plane_leader_value, max_plane_follower_value ,decision_rule = MILP_solve(beta_vector, self.belief_space.get_belief(belief_id), self.sota, self.gametype)

        # reconstruct alpha vectors  using optimal decision rule and beta-vector, then add them to the value function of respecive agents 
        leader_alpha,follower_alpha = reconstruct_alpha_vectors(belief_id,beta_vector,decision_rule)

        return leader_alpha,follower_alpha
    
    def backup(self,belief_id,timestep):
        "backup function for each belief id at a certain timestep"
        leader_alpha,follower_alpha  = self.solve(belief_id,timestep)
        self.vector_sets[timestep][belief_id] = JointAlphaVector(leader_alpha,follower_alpha,belief_id)

        self.leader_value_fn.add_alpha_vector(leader_alpha,timestep)
        self.folower_value_fn.add_alpha_vector(follower_alpha,timestep)
    
 
