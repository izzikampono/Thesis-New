import numpy as np
from beliefSpace import PROBLEM
from problem import PROBLEM
from decisionRule import DecisionRule
from jointAlphaVector import JointAlphaVector
from utilities import *
from alphaVector import AlphaVector
from betaVector import BetaVector
import sys

from utilities import PROBLEM
PROBLEM = PROBLEM.get_instance()
class MaxPlaneValueFunction:
    def __init__(self,belief_space,gametype,sota=False):
        #change this to initialize belief space here
        self.horizon = belief_space.horizon
        self.gametype = gametype
        self.belief_space = belief_space
        self.sota=sota
        # initialize value function
        self.vector_sets = {}
        self.point_value_fn = {}
        # initialize value functions to store alpha vectors of each timestep
        self.initialize_maxPlane_table()

       
#===== private methods ===============================================================================================

    def initialize_maxPlane_table(self):
        """ function to initialize the value function structure as a dictionary that stores alpha-vectors at each timestep"""
        for timestep in range(self.horizon+1):
            self.vector_sets[timestep] = {}
            self.point_value_fn[timestep] = {}
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.point_value_fn[timestep][belief_id] = None
                self.vector_sets[timestep][belief_id] = None

    def get_initial_value(self):
        """function that returns the tabular and max_plane value at timestep 0"""
        _, max_plane_value = self.get_max_alpha(belief_id=0,timestep=0)
        return max_plane_value , self.point_value_fn[0][0].get_value(self.belief_space.initial_belief)
    
    def get_vectors_at_timestep(self,timestep) -> JointAlphaVector:
        return self.vector_sets[timestep]

    def add_alpha_vector(self,alpha,timestep):
        """ function to store a new alpha-vector into the value function at a certain timestep """
        self.vector_sets[timestep][alpha.belief_id] = alpha
    
    def add_tabular_alpha_vector(self,alpha,timestep):
        """ function to store a new alph-vector into the tabular value function at a certain timestep """
        self.point_value_fn[timestep][alpha.belief_id] = alpha

    def get_max_alpha_mapping(self,belief_id,timestep,agent) -> JointAlphaVector:
        """function to get the corresponding maximum alpha-vector to a belief_id at a given timestep """
        if belief_id is not None: 
            #if the next timestep is the last horizon, then assign a 0 alpha vector
            if  timestep == self.horizon: return JointAlphaVector(None, AlphaVector(None,np.zeros(len(PROBLEM.STATES))), AlphaVector(None,np.zeros(len(PROBLEM.STATES))),belief_id)
            #else find the maximum alpha that maximizes the value corresponding to the belief resulting from (u,z)
            else:
                alpha, _  = self.get_max_alpha(belief_id, timestep,agent)
                return alpha
        else :
            return JointAlphaVector(None,None,None,None)
            # sys.exit()
    
    def get_tabular_mapping(self,next_belief_id,timestep,agent):
        # check if belief is valid
        if next_belief_id is not None: 
            # if next timestep is the last horizon, set the mapping to have a 0 alpha vector 
            if  timestep >= self.horizon: 
                return JointAlphaVector(None, AlphaVector(None,np.zeros(len(PROBLEM.STATES))), AlphaVector(None,np.zeros(len(PROBLEM.STATES))),next_belief_id)
            # else get the appropriate vector from the value function 
            else :
                return self.point_value_fn[timestep][next_belief_id]
        # else return placeholder vector
        else : return JointAlphaVector(None,None,None,None)
    
    def construct_beta(self,belief_id,mapping_belief_to_alpha) -> BetaVector:
        """function to construct beta vectors for a given subgame rooted at a belief_state.
           takes in a belief_state that the subgame is rooted at, and the mappings of future beliefs stemming from belief_state to its corresponding maximum alpha-vectors

            ## pseudo code :
            
            beta = zeros((X,U_joint))
            for x in X:
                for u in U_joint:
                    beta(x,u) = reward(x,u) 
                    for z in Z_joint:
                        for y in X :
                            beta(x,u) += dynamics(u,z,x,y) * next_optimal_alpha(u,z)[y]
        
        """
        #initialize beta
        beta_vectors = {0:np.zeros((len(PROBLEM.STATES),len(PROBLEM.JOINT_ACTIONS))),1:np.zeros((len(PROBLEM.STATES),len(PROBLEM.JOINT_ACTIONS)))}
        for agent in range(2):
            # if statement for the beta vector of blind opponents in SOTA =TRUE stackelberg games
            if self.gametype=="stackelberg" and self.sota == True and agent==1 :return BetaVector(beta_vectors[0],get_blind_beta())
            for state in PROBLEM.STATES :
                for joint_action in PROBLEM.JOINT_ACTIONS:
                    # beta(x,u) = reward(x,u)
                    beta_vectors[agent][state][joint_action] = PROBLEM.REWARDS[self.gametype][agent][joint_action][state]
                    for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                        # check (joint_action,joint_observation) branch that leads to the next optimal alpha vector from the perspective of the leader 
                        if self.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation) is not None:
                            for next_state in PROBLEM.STATES:
                                # beta(x,u) += \sum_{z} \sum_{y} DYNAMICS(u,z,x,y) * next_optimal_alpha(u,z)[y]
                                beta_vectors[agent][state][joint_action] += PROBLEM.TRANSITION_FUNCTION[joint_action][state][next_state] * PROBLEM.OBSERVATION_FUNCTION[joint_action][state][joint_observation] * mapping_belief_to_alpha[PROBLEM.LEADER][(joint_action,joint_observation)].individual_vectors[agent].vector[next_state]
        return BetaVector(beta_vectors[0],beta_vectors[1])
   

   
    def construct_max_plane_alpha_mapping(self,belief_id,timestep) -> dict:
        """function to create a mapping of all next_belief_ids branching from current_belief_id, to any alpha vector at horizon==timestep that gives the maximum value for each next_belief_id.
            
            ## psuedo code :

            for u in U_joint:
                for z in Z_joint:
                    map[(u,z)] = max_alpha(timestep+1,T(b_t,u,z))
        
        """
        #initialize mapping
        maxplane_alpha_mapping = {0:{},1:{}}
        # enumerate all joint_actions and joint_observations and get all next_belief_ids stemming from current belief_id
        for agent in range(2):
            for joint_action in PROBLEM.JOINT_ACTIONS:
                for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                    next_belief_id = self.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation)
                    # get alpha from next timstep that results in the maximum value for the current belief
                    maxplane_alpha_mapping[agent][(joint_action,joint_observation)] = self.get_max_alpha_mapping(next_belief_id,timestep+1,agent)
        return maxplane_alpha_mapping
        

    def construct_tabular_alpha_mapping(self,belief_id,timestep) -> dict:
        """ returns a mapping of next_belief_ids to alpha vectors. Next_belief_ids are a result of the enumaration of all possible beliefs by going through all joint actions and joint observations
        
            ## psuedo code :

            for u in U_joint:
                for z in Z_joint:
                    map[(u,z)] = value_function.vectors[t+1][T(b_t,u,z)]
        """

        # initialize mapping for both agents 
        tabular_belief_to_alpha_mapping ={0:{},1:{}}

        for agent in range(2):
            for joint_action in PROBLEM.JOINT_ACTIONS:
                for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                    # get next belief resulting from joint_action and joint_observation
                    next_belief_id = self.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation)
                    tabular_belief_to_alpha_mapping[agent][(joint_action,joint_observation)] = self.get_tabular_mapping(next_belief_id,timestep+1,agent)
        return tabular_belief_to_alpha_mapping

#===== public methods ===============================================================================================

    def print_initial_vector(self):
        alpha_0 = self.vector_sets[0][0]
        print(f"belief id = {alpha_0.belief_id},\tDR - joint {alpha_0.decision_rule.joint}\t- leader : {alpha_0.decision_rule.leader}\t- follower : {alpha_0.decision_rule.follower}")
        print(f"\nleader DR {alpha_0.get_follower_vector().decision_rule}\t, follower DR = {alpha_0.get_follower_vector().decision_rule}")

    def get_max_alpha(self,belief_id,timestep,agent=PROBLEM.LEADER)->tuple[JointAlphaVector,float] :
        """returns any alpha vector at timestep t that gives the maximum value for a given belief state"""
        #initialize max values 
        max = -np.inf
        max_alpha = None
        max_value = None
        belief = self.belief_space.get_belief(belief_id)
        # loop through all alpha-vectors at current timestep and find alpha-vector that gives maximum value
        for alpha in self.vector_sets[timestep].values():
            values = alpha.get_value(belief)
            # get the agent's value for this alpha at the current belief id and check if it is greater than currently stored "max" value
            if values[agent]>max :
                max = values[agent]
                max_value = values
                max_alpha = alpha
        return max_alpha,max_value
    
    
    def solve(self,belief_id,timestep) -> tuple[JointAlphaVector,JointAlphaVector]:
        """solve function that solves a subgame at each belief point in the tree by creating beta vectors and using a Mixed integer linear program to solve for the optimal decision rule"""
        # construct mapping of next_belief ids to optimal future alpha vectors
        alpha_mapping = self.construct_max_plane_alpha_mapping(belief_id, timestep)
        tabular_mapping =  self.construct_tabular_alpha_mapping(belief_id,timestep)
        
        # construct beta vectors
        max_plane_beta = self.construct_beta(belief_id, alpha_mapping)
        tabular_beta = self.construct_beta(belief_id,tabular_mapping)

        # solve subgame using mixed integer linear program (MILP)
        max_plane_leader_value, max_plane_follower_value ,max_plane_DR = MILP_solve(max_plane_beta, self.belief_space.get_belief(belief_id), self.sota, self.gametype)
        tabular_leader_value, tabular_follower_value ,tabular_DR = MILP_solve(tabular_beta, self.belief_space.get_belief(belief_id), self.sota, self.gametype)

        # reconstruct alpha vector using optimal decision rule and beta 
        max_plane_alpha = reconstruct_alpha_vectors(belief_id,max_plane_beta,max_plane_DR)
        tabular_alpha = reconstruct_alpha_vectors(belief_id,tabular_beta,tabular_DR)

        # add mapping to reconstructed alphas 
        max_plane_alpha.set_map(alpha_mapping)
        tabular_alpha.set_map(tabular_mapping)

        ## print statements for verification 
        if np.abs(max_plane_leader_value - tabular_leader_value)> 0.000001 :
            print(f"found difference in LP : maxplane = {max_plane_leader_value}, tabular = {tabular_leader_value}")
            print(f"Decision rules ::\n\t Max-plane =\n {max_plane_alpha.decision_rule.joint},\n\ttabular =\n {tabular_alpha.decision_rule.joint}")
            print(f"running verification for belief {belief_id} at timestep {timestep}")
             #if the tabular value is different from the max_plane value, then print and terminate 
            # print(f"max-plane alpha = {max_plane_alpha.individual_vectors}")
            print(f"\t\treconstructed max-plane alpha {max_plane_alpha.get_vectors()},\ttabular alpha {tabular_alpha.get_vectors()} ")

            for joint_action in PROBLEM.JOINT_ACTIONS:
                print(f"\t\t\t joint_action:{joint_action}")
                for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                    next_max_plane_alpha = alpha_mapping[PROBLEM.LEADER][(joint_action,joint_observation)]
                    next_belief_id = self.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation)
                    tabular_next_alpha = tabular_mapping[PROBLEM.LEADER][(joint_action,joint_observation)]
                    if next_max_plane_alpha is not None: 
                        next_tabular_value = tabular_next_alpha.get_value(self.belief_space.get_belief(next_belief_id))
                        next_maxplane_value = next_max_plane_alpha.get_value(self.belief_space.get_belief(next_belief_id))
                        print(f"\t\t\t\t joint_observation:{joint_observation} proba:{observation_probability(joint_observation, self.belief_space.get_belief(belief_id), joint_action)}, maxplane value = {next_maxplane_value} , tabular value = {next_tabular_value}")

                        # print(f"\t\t\t\t\tbelief: {next_belief_id} = {self.belief_space.get_belief(next_belief_id).value} , \tmaxplane-value: {next_max_alpha_value} / {next_max_alpha.vectors} , \ttabular-value:  {next_tabular_value}  / {self.point_value_fn[timestep+1][next_belief_id].vectors}")
                        if np.abs(next_maxplane_value[0] - next_tabular_value[0])> 0.00001 :
                            print(f"\t\t\t\t\tbelief: {next_belief_id} = {self.belief_space.get_belief(next_belief_id).value},\tmaxplane-value built on belief {next_max_plane_alpha.belief_id}: {next_maxplane_value},\ttabular-value:  {next_tabular_value}  ")
                            print(f"\t\t\t\t\tAlpha Vectors :\n\t\t\t\t\t max_plane alpha = {next_max_plane_alpha.get_vectors()},\n\t\t\t\t\ttabular alpha = {tabular_next_alpha.get_vectors()} ")
            
            print(f"\t\t\t ====================================")


            for agent in range(2):
                for state in PROBLEM.STATES:
                    for joint_action in PROBLEM.JOINT_ACTIONS:
                        reward = PROBLEM.REWARDS[self.gametype][agent][joint_action][state]
                        if np.abs(max_plane_beta.two_d_vectors[agent][state][joint_action] -  tabular_beta.two_d_vectors[agent][state][joint_action]) >0:
                            print(f"\t\t\tbeta({state},{joint_action}) :: reward = {reward} max-plane {max_plane_beta.two_d_vectors[agent][state][joint_action]-reward}, \t tabular = {tabular_beta.two_d_vectors[agent][state][joint_action]-reward}")


            sys.exit()

        return max_plane_alpha, tabular_alpha
    
    def backup(self,belief_id,timestep):
        "backup function for each belief id at a certain timestep"
        max_plane_alpha, tabular_alpha  = self.solve(belief_id,timestep)
        self.add_alpha_vector(max_plane_alpha,timestep)
        self.add_tabular_alpha_vector(tabular_alpha,timestep)

    def verify(self,belief_id,timestep):
        """ function that is called after the backup of all belief states at a certain timestep in order to verify if the backup resulted in consistent values between the max_plane and tabular algorithms.
            the function prints out details in the terminal if there are any inconsistencies found.
        """

        # get current belief state, and its corresponding tabular and maxplane values
        belief = self.belief_space.get_belief(belief_id)
        tabular_value = self.point_value_fn[timestep][belief_id].get_value(belief)
        max_alpha, max_alpha_value = self.get_max_alpha(belief_id,timestep)
        
        # construct mapping of next_belief ids to optimal future alpha vectors
        alpha_mapping = self.construct_max_plane_alpha_mapping(belief_id, timestep)
        tabular_mapping =  self.construct_tabular_alpha_mapping(belief_id,timestep)
        
        
        print(f"\t\tbelief: {belief_id} = {belief.value} , \tmaxplane-value: {max_alpha_value}, \ttabular-value:  {tabular_value}")
        
      

        #if the tabular value is different from the max_plane value, then print and terminate 
        if np.abs(tabular_value[0] - max_alpha_value[0])> 0.0001 :
            print(f"\t\t========= FOUND DIFFERENCE IN VERIFICATION! ========")
            print(f"\t\tmax-plane alpha {max_alpha.get_vectors()} from belief id {max_alpha.belief_id}, tabular alpha {self.point_value_fn[timestep][belief_id].get_vectors()} ")


            for joint_action in PROBLEM.JOINT_ACTIONS:
                print(f"\t\t\t joint_action:{joint_action}")
                for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                    print(f"\t\t\t\t joint_observation:{joint_observation} proba:{observation_probability(joint_observation, belief, joint_action)}")
                    next_belief_id = self.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation)
                    if next_belief_id is not None and timestep+1<= self.horizon: 
                        next_tabular_alpha_from_mapping = tabular_mapping[PROBLEM.LEADER][(joint_action,joint_observation)]
                        next_tabular_alpha_value = next_tabular_alpha_from_mapping.get_value(self.belief_space.get_belief(next_belief_id))
                        next_max_alpha_from_mapping = alpha_mapping[PROBLEM.LEADER][(joint_action,joint_observation)]
                        next_max_alpha_from_mapping_value = next_max_alpha_from_mapping.get_value(self.belief_space.get_belief(next_belief_id))

                        # next_max_alpha, next_max_alpha_value = self.get_max_alpha(next_belief_id, timestep+1)

                        if np.abs(next_max_alpha_from_mapping_value[0] - next_tabular_alpha_value[0])> 0.0001 :
                            print(f"\t\t\t\t\tbelief: {next_belief_id} = {self.belief_space.get_belief(next_belief_id).value},\tmaxplane-value built on belief {next_max_alpha_from_mapping.belief_id}: {next_max_alpha_from_mapping_value},\ttabular-value:  {next_tabular_alpha_value}  ")
                            print(f"\t\t\t\t\tAlpha Vectors :\n\t\t\t\t\t\t max_plane alpha = {next_max_alpha_from_mapping.get_vectors()},\n\t\t\t\t\t\ttabular alpha ={self.point_value_fn[timestep+1][next_belief_id].get_vectors()} ")
                            print(f"\t\t\t\ max alpha vector on actual belief id = {self.vector_sets[timestep+1][next_belief_id].get_vectors()}, ")

            print(f"\t\t=====================================================")

  

  