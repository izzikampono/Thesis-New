import numpy as np
from alphaVector import PROBLEM, AlphaVector
from jointAlphaVector import JointAlphaVector
from valueFunction import ValueFunction
from beliefSpace import PROBLEM, BeliefSpace
from decisionRule import DecisionRule
from problem import PROBLEM
import time
import copy
import sys
from utilities import *
PROBLEM = PROBLEM.get_instance()
import gc
gc.enable()

class  PBVI:
    """class to represent the instance of a game to be solved using either the maxplane or tabular mode"""
    def __init__(self,horizon,gametype,belief_space):
        self.horizon = horizon
        self.gametype = gametype
        self.value_function = ValueFunction(horizon,belief_space,self.gametype,None)


#===== private methods ===============================================================================================


    def reset_value_function(self,belief_space):
        """function to reset belief space and value function before solving a new/different game"""
        # set new density if the optional input to the function is given
        self.value_function = ValueFunction(self.horizon,belief_space,self.gametype,None)

    
    def backward_induction(self):
        """function goes through all belief_states in the range of the planning horizon and conducts the backup operator on each belief_state"""
       
        # start loop with timestep at horizon-1 and iterate to timestep 0
        for timestep in range(self.horizon-1,-1,-1):
            print(f"\n========== Backup at timestep {timestep} ==========")
            # loop through all beliefs at a given timestep
            n = 1
            belief_list = self.value_function.belief_space.time_index_table[timestep]
            for belief_id in belief_list:
                print(f"\t\tbelief id : {belief_id} - {n} / {len(self.value_function.belief_space.time_index_table[timestep])} ")
                self.value_function.backup(belief_id,timestep)

                n+=1
        
        # terminal result printing
        leader_value , follower_value = self.value_function.get_initial_value()
        print(f"\n\n\n================================================= END OF {self.gametype} GAME WITH SOTA {self.value_function.sota} ================================================================")
        print(f"\n\t\t\t\t alphavectors value at inital belief (V0,V1) : leader =  { leader_value} , follower = {follower_value}")
        print(f"\n==========================================================================================================================================================================")
        return leader_value , follower_value

    def print_leader_policy(self,joint_alpha: JointAlphaVector ,timestep):
        if timestep>= self.horizon : return
        print("∟ DR : ",joint_alpha.individual_vectors[PROBLEM.LEADER].decision_rule)
        for joint_action in PROBLEM.JOINT_ACTIONS:
            for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                print(self.value_function.belief_space.existing_next_belief_id(joint_alpha.belief_id,joint_action,joint_observation))
                self.print_leader_policy(joint_alpha.individual_vectors[PROBLEM.LEADER].get_future_alpha(joint_action,joint_observation),timestep+1)

    def print_follower_policy(self,joint_alpha: JointAlphaVector ,timestep):
        if timestep>= self.horizon : return
        print("∟ DR : ",joint_alpha.individual_vectors[PROBLEM.FOLLOWER].decision_rule)
        for joint_action in PROBLEM.JOINT_ACTIONS:
            for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                print(self.value_function.belief_space.existing_next_belief_id(joint_alpha.belief_id,joint_action,joint_observation))
                self.print_leader_policy(joint_alpha.individual_vectors[PROBLEM.FOLLOWER].get_future_alpha(joint_action,joint_observation),timestep+1)

#===== public methods ===============================================================================================

    def solve_game(self,sota):
        "solve function that solves 1 iteration of a game using a fixed density"

        # set solve method (state of the art or stackelberg):
        self.value_function.sota = sota
        start_time = time.time()
        #expand belief with desired density
        # self.value_function.belief_space.expansion()

        # conduct backward induction of all horizons
        values = self.backward_induction()
        
        # return policy, value at the initial belief and the time it took to solve the game 
        return values, time.time() - start_time , self.value_function.belief_space.size_at_horizon(self.horizon)
   
  
    
    def evaluate(self,belief_id,timestep,leader_value_fn: ValueFunction,follower_value_fn:ValueFunction) -> tuple[float,float]:
        """function to evaluate leader and follower poliicies from different solve methods (stackelberg/state of the are).
           Does this by doing another point based backup using belief points from the last stage up to the 0th stage.
           at each stage, it uses decision rules prescribed by the input policy to create an alphavector representative of the solution of the subgame rooted at that belief_state .         
        """

        """ pseudo code :: 

            for t in {H-1,..,0} :
                for b in beliefspace[t] :
                    for x in X :
                        value = 0
                        for u_joint in U_joint:
                            value += r(x,u_joint) * a(u_joint)
                            for z in Z:
                                next_b = Transition(b,u,z)
                                for y in X:
                                    value += Dynamics(y,z,x,u) * V_{t+1}(next_b)[y] * u_joint_probability
                            
        """

        # initialize table of values and get all belief values by traversing the tree in a bottom up manner
        value_fn = {}
        for timestep in range(self.horizon-1,-1,-1):
            value_fn[timestep] = {}
            print(f"timestep = {timestep}")
            for belief_id in self.value_function.belief_space.time_index_table[timestep]:
                # initialize value function to store alpha vector for each agent at each beleif point
                value_fn[timestep][belief_id] = {PROBLEM.LEADER:{},PROBLEM.FOLLOWER:{}}
                for agent in range(2):
                    for state in PROBLEM.STATES:

                        state_value = 0

                        for joint_action in PROBLEM.JOINT_ACTIONS:
                          
                            # seperate the join action into seperate agent actions and get the probabilitiy of each action from the two seperate policies 
                            leader_action, follower_action = PROBLEM.get_seperate_action(joint_action)
                            joint_action_probability = leader_value_fn.get_leader_DR(belief_id,timestep)[leader_action] * follower_value_fn.get_follower_DR(belief_id,timestep)[state][follower_action]
                            
                            # get reward component :: r(x,u) * a(u_joint) * b(x)
                            state_action_value = PROBLEM.REWARDS["generalsum"][agent][joint_action][state]  * self.value_function.belief_space.get_belief(belief_id).value[state] * joint_action_probability
                            
                            if timestep+1 < self.horizon:
                                for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                                    # next_belief = Transtition(b,u,z)
                                    next_belief_id,flag = self.value_function.belief_space.existing_next_belief_id(belief_id,joint_action,joint_observation,timestep)
                                    if next_belief_id is not None:
                                        for next_state in PROBLEM.STATES :
                                       
                                            # get future component :: Pr(y,z|x,u) * V_t+1(next_b)[y] * a_joint(u_joint|x)
                                            state_action_value +=  joint_action_probability * PROBLEM.TRANSITION_FUNCTION[joint_action][state][next_state] * PROBLEM.OBSERVATION_FUNCTION[joint_action][state][joint_observation] *  value_fn[timestep+1][next_belief_id][agent][next_state]
                                       
                           
                            # add current state_action value to cummulative state_value
                            state_value += state_action_value
                        
                        # add alphavector values in the value function by state 
                        value_fn[timestep][belief_id][agent][state] = state_value
                    
                print(f"\tbelief = {belief_id},\tvalue :: {value_fn[timestep][belief_id]}")
        
        ## get value at initial belief ::
        initial_value = np.zeros(2)
        for agent in range(2):
            for state in PROBLEM.STATES: 
                initial_value[agent] += self.value_function.belief_space.initial_belief.value[state] * value_fn[0][0][agent][state]

        return initial_value
        

        