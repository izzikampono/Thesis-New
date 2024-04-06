
from problem import PROBLEM
import numpy as np
import temp_problem
import pytest

# sert - compares the result of the Act with the desired result 


from utilities import PROBLEM
class testBelief:
    """Mock class that immitates a beta function used for the testing pipeline"""
    def __init__(self) -> None:
        self.value = np.linspace(0,1,len(PROBLEM.STATES))


# make a mock instance of the class as a fixture so it can be used as an argument of test function
@pytest.fixture  
def temp_belief_state():
    return np.linspace(0,1,len(PROBLEM.STATES))

def next_belief(belief,joint_action,joint_observation):
    # initialize next_belief value
    next_belief_value= np.zeros(len(PROBLEM.STATES))

    # calculate next belief state with b_next(x') = \sum_{x} += b_current(x) * TRANSITION_MATRIX(u,z,x,x')
    for next_state in PROBLEM.STATES:
        value = 0
        for state in PROBLEM.STATES:
            value += belief[state] * PROBLEM.TRANSITION_FUNCTION[joint_action][state][next_state]  * PROBLEM.OBSERVATION_FUNCTION[joint_action][state][joint_observation]
        next_belief_value[next_state]+=value  

    next_belief_value = np.array(next_belief_value) / np.sum(next_belief_value)
    return next_belief_value


def test_next_belief(temp_belief_state):
    joint_action = 0
    joint_observation = 0
    assert sum(next_belief(temp_belief_state,joint_action,joint_observation)) == 1

