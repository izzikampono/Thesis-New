import numpy as np 
import sys
from utilities import normalize,observation_probability
from problem import PROBLEM
PROBLEM = PROBLEM.get_instance()

class BeliefState:
    """Class that represents a Belief State object in a given game

        Attributes ::
        - value = vector of belief values over states x \in X
        - id = the id that corresponds to this BeliefState object, the id is used to keep track of belief states object in the Belief Space
       
    
    
    
    """
    def __init__(self,value,action_label=None,observation_label=None,id=None):
        self.value = value
        self.id = id
        # self.action_label = action_label
        # self.observation_label = observation_label

    def set_id(self,id):
        self.id=id
    
    def check_validity(self,next_belief_value):
        """function to check that a given belief_state does not sum up to one """
        if np.sum(next_belief_value) == 0 :
            print("found all zero belief")
            sys.exit()

    def normalize(self,next_belief_value):
        # if next_belief is valid (does not sum to 0 ), we normalize to make it sum up to one
        next_belief_value = normalize(next_belief_value)
        # extra check
        if np.sum(next_belief_value)<= 1.001 and np.sum(next_belief_value)> 0.99999:
            return next_belief_value
        else:
            print("err0r : belief doesn not sum up to 1\n")
            print(f"current belief: \n{self.value}")
            print(f"next belief :\n{next_belief_value}")
            print(f"sum : {np.sum(next_belief_value)}")
            sys.exit()

    def next_belief(self,joint_action,joint_observation) :
        """function to calculate next belief based on current belief based on the joint action and joint observation. returns a new belief state
        
            ## pseudo code to calculate next belief :
            b_next = np.zeros(len(PROBLEM.STATES))
            for x in X:
                value = 0
                for y in X :
                    value += b(x) * dynamics(u,z,x,y)
                b_next[x] = value        
        
        """
        
        # initialize next_belief value
        next_belief_value= np.zeros(len(PROBLEM.STATES))

        # calculate next belief state with b_next(x') = \sum_{x} += b_current(x) * TRANSITION_MATRIX(u,z,x,x')
        for next_state in PROBLEM.STATES:
            # initialize b'(x)
            value = 0
            for state in PROBLEM.STATES:
                # b'(x) += b(x) * dynamics(x,u,y,z)
                value += self.value[state] * PROBLEM.TRANSITION_FUNCTION[joint_action][state][next_state]  * PROBLEM.OBSERVATION_FUNCTION[joint_action][state][joint_observation]
            next_belief_value[next_state]+=value  

        # check if the next_belief is not valid 
        self.check_validity(next_belief_value)
        # normalize
        next_belief_value = self.normalize(next_belief_value)
        return BeliefState(next_belief_value,joint_action,joint_observation)
    

    def coniditioned_observation_distribution(self,joint_action):
        """builds the probability distribution of observations conditioned on the current belief state and joint action"""
        observation_distribution = []
        for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
            observation_distribution.append(observation_probability(joint_observation,self,joint_action))
        return np.array(observation_distribution)

    def print(self):
        print(self.value)