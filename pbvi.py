import numpy as np
from alphaVector import PROBLEM, AlphaVector
from jointAlphaVector import JointAlphaVector
from valueFunction import ValueFunction
from beliefSpace import PROBLEM, BeliefSpace
from decisionRule import DecisionRule
from problem import PROBLEM
import time
from utilities import *
PROBLEM = PROBLEM.get_instance()
import gc
gc.enable()

class  PBVI:
    """class to represent the instance of a game to be solved using either the maxplane or tabular mode"""
    def __init__(self,horizon,gametype,sota,density):
        self.horizon = horizon
        self.gametype = gametype
        self.sota = sota
        self.density = density
        self.belief_space = BeliefSpace(horizon,self.density)
        self.value_function = ValueFunction(self.belief_space,self.gametype,self.sota)


#===== private methods ===============================================================================================


    def reset(self,density=None):
        """function to reset belief space and value function before solving a new/different game"""
        # set new density if the optional input to the function is given
        if density is not None: self.density = density
        self.belief_space.reset(self.density)
        self.value_function = ValueFunction(self.belief_space,self.gametype,self.sota)
    
    def backward_induction(self):
        """function goes through all belief_states in the range of the planning horizon and conducts the backup operator on each belief_state"""
       
        # start loop with timestep at horizon-1 and iterate to timestep 0
        for timestep in range(self.horizon-1,-1,-1):
            print(f"\n========== Backup at timestep {timestep} ==========")
            # loop through all beliefs at a given timestep
            n = 1
            for belief_id in self.belief_space.time_index_table[timestep]:
                self.value_function.backup(belief_id,timestep)
                print(f"\t\tbelief id : {belief_id} - {n} / {len(self.belief_space.time_index_table[timestep])} ")
                n+=1
        
        # terminal result printing
        leader_value , follower_value = self.value_function.get_initial_value()
        print(f"\n\n\n================================================= END OF {self.gametype} GAME WITH SOTA {self.sota} ================================================================")
        print(f"\n\t\t\t\t alphavectors value at inital belief (V0,V1) : leader =  { leader_value} , follower = {follower_value}")
        print(f"\n==========================================================================================================================================================================")
        return leader_value , follower_value

    def print_leader_policy(self,joint_alpha: JointAlphaVector ,timestep):
        if timestep>= self.horizon : return
        print("∟ DR : ",joint_alpha.individual_vectors[PROBLEM.LEADER].decision_rule)
        for joint_action in PROBLEM.JOINT_ACTIONS:
            for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                print(self.belief_space.existing_next_belief_id(joint_alpha.belief_id,joint_action,joint_observation))
                self.print_leader_policy(joint_alpha.individual_vectors[PROBLEM.LEADER].get_future_alpha(joint_action,joint_observation),timestep+1)

    def print_follower_policy(self,joint_alpha: JointAlphaVector ,timestep):
        if timestep>= self.horizon : return
        print("∟ DR : ",joint_alpha.individual_vectors[PROBLEM.FOLLOWER].decision_rule)
        for joint_action in PROBLEM.JOINT_ACTIONS:
            for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                print(self.belief_space.existing_next_belief_id(joint_alpha.belief_id,joint_action,joint_observation))
                self.print_leader_policy(joint_alpha.individual_vectors[PROBLEM.FOLLOWER].get_future_alpha(joint_action,joint_observation),timestep+1)

#===== public methods ===============================================================================================


    def extract_policy(self):
        """public function to extract policies of both agents after a game is solved"""
        print("Extracting Policies..")
        self.value_function.get_max_alpha(0,0,PROBLEM.LEADER)
        self.value_function.get_max_alpha(0,0,PROBLEM.FOLLOWER)

        policies = [self.extract_leader_policy(belief_id=0,timestep=0),  self.extract_follower_policy(belief_id=0,timestep=0)]    

        return policies
    
    def solve_game(self,density=None):
        "solve function that solves 1 iteration of a game using a fixed density"

        #reset belief space and value function before each solve 
        self.reset(density)
        start_time = time.time()
        #expand belief with desired density
        self.belief_space.expansion()

        #conduct backward induction of all horizons
        values = self.backward_induction()
        
        # return policy, value at the initial belief and the time it took to solve the game 
        return values, time.time() - start_time
        
    def evaluate(self,belief_id,timestep,leader_alpha: AlphaVector,follower_alpha:AlphaVector) -> tuple[float,float]:
        """recursive function to get the values from two seperate individual policies, the function traverses individuals policies in parallel to get a joint value for the game.
            this function calculates the joint value by simulating a game where the agents follow the prescribed policies from different solve methods (Stackelberg/State of the art)
        """
        # edge case of the recursive function
        if  timestep == self.horizon or leader_alpha is None or follower_alpha is None: return (0,0)
        
        #initialize values list and get belief state corresponding to belief_id
        values = []
        belief = self.belief_space.get_belief(belief_id)
        
        
        # get V(b) recursively by \sum_{x} \sum{u_joint} += b(x) * leader_decision_rule(u1) *  follower_decision_rule(u2) + \sum_{z} += Pr(z|b,u_joint) * V(TRANSITION(b,u_joint,z))
        for agent in range(2):
            value = 0
            reward = PROBLEM.REWARDS["stackelberg"][agent]
            for state in PROBLEM.STATES:
                for follower_action in PROBLEM.ACTIONS[PROBLEM.FOLLOWER]:
                    for leader_action, leader_action_probability in enumerate(leader_alpha.decision_rule):
                        # check if action probabilities are greater than 0
                        if follower_alpha.decision_rule[state][follower_action]>0 and leader_action_probability>0:
                            joint_action = PROBLEM.get_joint_action(leader_action,follower_action)
                            # get value of current stage of the game :: value = b(x) * a1(u1) * a2(u2) * reward((u1,u2),x)
                            value += belief.value[state] * leader_alpha.decision_rule[leader_action] * follower_alpha.decision_rule[state][follower_action] * reward[joint_action][state]
                            for joint_observation in PROBLEM.JOINT_OBSERVATIONS:

                                next_belief_id = self.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation)
                                
                                # get value of future stages of the game :: value += Pr(z|b,u) * evaluate(timestep+1,next_b)
                                if next_belief_id is not None and timestep+1<self.horizon: 
                                    next_alpha_pair = self.value_function.get_alpha_pairs(timestep+1,next_belief_id)
                                    value +=  observation_probability(joint_observation,belief,joint_action) * self.evaluate(belief_id, timestep+1, next_alpha_pair.get_leader_vector() , next_alpha_pair.get_follower_vector())[agent]
            values.append(value)
        return values
        

        