
import numpy as np
from problem import PROBLEM
from alphaVector import AlphaVector
from beliefSpace import BeliefSpace
from utilities import *
PROBLEM = PROBLEM.get_instance()
import gc
gc.enable()


class FollowerValueFunction:
    def __init__(self,belief_space:BeliefSpace,gametype,sota) :
        self.belief_space = belief_space
        self.horizon = belief_space.horizon
        self.gametype = gametype
        self.sota=sota
        self.initialize_value_function()


    def initialize_value_function(self):
        self.vector_sets = {}
        for timestep in range(self.horizon+1):
            self.vector_sets[timestep] = {}
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.vector_sets[timestep][belief_id] = None

    def get_initial_value(self):
        """function to get the value of the alpha vector at the initial belief"""
        return self.get_vector_at_belief(0,0).vector
        
    
    def get_vectors_at_timestep(self,timestep) -> list[AlphaVector]:
        """function to return list of alpha vectors that have been stored in the value function"""
        return self.vector_sets[timestep]
    
     
    def get_vector_at_belief(self,belief_id,timestep) -> AlphaVector:
        return self.vector_sets[timestep][belief_id]

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
        follower_beta = np.zeros((len(PROBLEM.STATES),len(PROBLEM.JOINT_ACTIONS)))
        # if statement for blind follower agent 
        if self.sota==True : return self.construct_blind_beta()
        
        for state in PROBLEM.STATES :
            for joint_action in PROBLEM.JOINT_ACTIONS:
                # beta(x,u) = reward(x,u)
                follower_beta[state][joint_action] = PROBLEM.REWARDS[self.gametype][PROBLEM.FOLLOWER][joint_action][state]
                
                # no future value needed if current timestep = last timestep
                if timestep+1 >= self.horizon : continue
                
                for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                    next_belief_id = self.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation) 
                    # check (joint_action,joint_observation) branch that leads to the next optimal alpha vector from the perspective of the leader 
                    if next_belief_id is not None:
                        for next_state in PROBLEM.STATES:
                            # beta(x,u) += \sum_{z} \sum_{y} DYNAMICS(u,z,x,y) * next_optimal_alpha(u,z)[y]
                            follower_beta[state][joint_action] += PROBLEM.TRANSITION_FUNCTION[joint_action][state][next_state] * PROBLEM.OBSERVATION_FUNCTION[joint_action][state][joint_observation] * self.vector_sets[timestep+1][next_belief_id].vector[state]
        return follower_beta
    
    def construct_blind_beta(self,):
        """function that constructs the beta vectors of the blind follower.
           The blind follower only uses the immediate reward from a given state action pair to estimate its value
        """
        follower_beta = np.zeros((len(PROBLEM.STATES),len(PROBLEM.JOINT_ACTIONS)))
        for state in PROBLEM.STATES :
            for joint_action in PROBLEM.JOINT_ACTIONS:
                # beta(x,u) = reward(x,u)
                follower_beta[state][joint_action] = PROBLEM.REWARDS[self.gametype][PROBLEM.FOLLOWER][joint_action][state]
        return follower_beta



    def add_alpha_vector(self,alpha,timestep):
        """ function to store a new alpha-vector into the value function at a certain timestep """
        self.vector_sets[timestep][alpha.belief_id] = alpha

        