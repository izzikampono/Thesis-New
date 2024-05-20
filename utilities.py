from __future__ import barry_as_FLUFL
from ast import Tuple
import warnings
import numpy as np 
import sys
from problem import PROBLEM
from alphaVector import AlphaVector
from SOTAStrategy import *


"""This module defines some static utility functions to be used throughout the code."""


__all__ = ["normalize","observation_probability","get_blind_beta","MILP_solve","reconstruct_alpha_vectors","exponential_decrease"]
__version__ = "0.1.0"

PROBLEM = PROBLEM.get_instance()


def normalize(vector):
    """function to normalize a vector"""
    warnings.filterwarnings("error", category=RuntimeWarning)
    try:
        vector = np.array(vector) / np.sum(vector)
        return vector
    except RuntimeWarning as rw:
        print(f"RuntimeWarning: {rw}")
        print(f"cannot normalize vector V: {vector}")
        sys.exit()



def observation_probability(joint_observation,belief,joint_action):
    """function to calculate probability of an observation given a belief and joint action
    
        Pseudo-code :

        observation_probability = 0
        for x in X:
            for y in X:
                observation_probability += b(x) * Transition_fn(u,x,y) * Observation_fn(u,x,z)     
    """

    sum=0
    for state in PROBLEM.STATES:
        for next_state in PROBLEM.STATES:
                sum += belief.value[state]  * PROBLEM.TRANSITION_FUNCTION[joint_action][state][next_state] * PROBLEM.OBSERVATION_FUNCTION[joint_action][state][joint_observation]
    return sum

def get_blind_beta():
    """ build beta vector for blind opponent (only uses current reward without an expectation of future rewards """

    reward = PROBLEM.REWARDS["stackelberg"]
    two_d_vector = np.zeros((len(PROBLEM.STATES),len(PROBLEM.JOINT_ACTIONS)))

    for state in PROBLEM.STATES:
        for joint_action in PROBLEM.JOINT_ACTIONS:
            # get reward of follower agent for each action,state value
            two_d_vector[state][joint_action] = reward[PROBLEM.LEADER][joint_action][state]
    return two_d_vector

def get_joint_DR(leader_decision_rule,follower_decision_rule):
    """function to caculate the joint decision rule from individual decision rules
        for x in X:
            for u1 in U1:
                for u2 in U2 :
                    joint_DR = a1(u1) * a2(u2)

    """

    # initialize
    joint_decision_rule = {}

    for state in PROBLEM.STATES:
        joint_decision_rule[state] = np.zeros(len(PROBLEM.JOINT_ACTIONS))
        for leader_action,leader_action_probability in enumerate(leader_decision_rule):
            for follower_action,follower_action_probability in enumerate(follower_decision_rule[state]):
                # joint DR ::  \sum_{x} \sum_{u_joint} += a_1(u_1) * a_2(u_2|x)
                joint_decision_rule[state][PROBLEM.get_joint_action(leader_action,follower_action)] = leader_action_probability * follower_action_probability
    return joint_decision_rule




def MILP_solve(beta,belief,sota,gametype):
    """function that creates the main pipeline of which linear program to use depending on the gametype (common reward/general sum/zerosum) and the solve mode chosen (Stackelberg/state of the art (sota))"""
    
    # initialize solution variables
    leader_value , follower_value ,decision_rule = None, None, None
   
    ## main pipeline
    # if solve mode is Stackelberg
    if sota==False:
        # use stackelberg mix integer linear program
        milp = MILP(belief,beta)
        return milp.run()
    
    # else use the State of the art strategies of each gametype
    else:
        if gametype=="zerosum":
            return zerosum_sota(belief,beta)
        if gametype=="generalsum":
            return stackelberg_sota(belief,beta)
        if gametype=="cooperative":
            return cooperative_sota(belief,beta)
    return leader_value,follower_value,decision_rule



def reconstruct_alpha_vectors(belief_id, beta, optimal_decision_rule) ->tuple[AlphaVector]:
    "function to reconstruct an alpha vector by using the beta constructed on the belief_id and the optimal decision rule from the linear programs "
    vectors = np.zeros((2,len(PROBLEM.STATES)))
    for state in PROBLEM.STATES:
        for leader_action in PROBLEM.ACTIONS[PROBLEM.LEADER]:
            for follower_action in PROBLEM.ACTIONS[PROBLEM.FOLLOWER]:
                joint_action = PROBLEM.get_joint_action(leader_action,follower_action)
                vectors[PROBLEM.LEADER][state] += optimal_decision_rule.leader[leader_action] * optimal_decision_rule.follower[state][follower_action]  * beta.two_d_vectors[PROBLEM.LEADER][state][joint_action] 
                vectors[PROBLEM.FOLLOWER][state] +=optimal_decision_rule.leader[leader_action] * optimal_decision_rule.follower[state][follower_action]  * beta.two_d_vectors[PROBLEM.FOLLOWER][state][joint_action] 
    return AlphaVector(optimal_decision_rule.leader,vectors[PROBLEM.LEADER],belief_id) , AlphaVector(optimal_decision_rule.follower,vectors[PROBLEM.FOLLOWER],belief_id)

def exponential_decrease(starting_value, num,factor = 5):
    "function to generate decreasing values to use as the density (hyperparameter) values for the expansion of the belief space. "
    densities = [starting_value]
    while len(densities)<num:
        starting_value /= factor
        densities.append(starting_value)
    return densities