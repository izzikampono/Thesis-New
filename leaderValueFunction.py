
import time
import numpy as np
from alphaVector import AlphaVector
from betaVector import BetaVector
from problem import PROBLEM
PROBLEM = PROBLEM.get_instance()
from utilities import *
import gc
gc.enable()


class LeaderValueFunction:
    def __init__(self,belief_space,gametype,sota) :
        self.belief_space = belief_space
        self.horizon = belief_space.horizon
        self.gametype = gametype
        self.sota=sota
        self.vector_sets = {}
        self.initialize_value_function()


    def initialize_value_function(self):
        """function that initializes a 2D dictionary to store all alpha vectors.
           the dictionary entitles "self.vector_sets" stores the alphavecors and is indexable along 2 axis : timestep and belief_id
        
        """
        for timestep in range(self.horizon+1):
            self.vector_sets[timestep] = {}
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.vector_sets[timestep][belief_id] = None
    
    def get_initial_value(self):
        """function that returns the value of the alphavector at timestep 0, when evaluated using the initial belief state b0
           returns  ::  \sum alpha_0(x) * b_0(x)
        """
        return self.get_vector_at_belief(0,0).get_value(self.belief_space.get_belief(0))
    
    def get_vectors_at_timestep(self,timestep) -> AlphaVector:
        return self.vector_sets[timestep]
    
    def get_vector_at_belief(self,belief_id,timestep) -> AlphaVector:
        return self.vector_sets[timestep][belief_id]

    def add_alpha_vector(self,alpha,timestep):
        """ function to store a new alpha-vector into the value function at a certain timestep """
        self.vector_sets[timestep][alpha.belief_id] = alpha

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
        leader_beta = np.zeros((len(PROBLEM.STATES),len(PROBLEM.JOINT_ACTIONS)))
        for state in PROBLEM.STATES :
            for joint_action in PROBLEM.JOINT_ACTIONS:
                # beta(x,u) = reward(x,u)
                leader_beta[state][joint_action] = PROBLEM.REWARDS[self.gametype][PROBLEM.LEADER][joint_action][state]

                if timestep+1 >= self.horizon : 
                    continue

                for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                    next_belief_id = self.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation) 
                    # check (joint_action,joint_observation) branch that leads to the next optimal alpha vector from the perspective of the leader 
                    if next_belief_id is not None:
                        for next_state in PROBLEM.STATES:
                            # beta(x,u) += \sum_{z} \sum_{y} DYNAMICS(u,z,x,y) * next_optimal_alpha(u,z)[y]
                            leader_beta[state][joint_action] += PROBLEM.TRANSITION_FUNCTION[joint_action][state][next_state] * PROBLEM.OBSERVATION_FUNCTION[joint_action][state][joint_observation] * self.vector_sets[timestep+1][next_belief_id].vector[state]
        return leader_beta