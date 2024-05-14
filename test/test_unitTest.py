
from alphaVector import AlphaVector
from beliefState import BeliefState
from beliefSpace import BeliefSpace
from betaVector import BetaVector
from jointAlphaVector import JointAlphaVector
from decisionRule import DecisionRule
from betaVector import BetaVector


def is_beliefState(belief_state):
    assert type(belief_state) == BeliefState

def is_alphaVector(alpha):
    assert type(alpha) == AlphaVector

def is_betaVector(beta):
    assert type(beta) == BetaVector

def is_jointAlphaVector(alpha):
    assert type(alpha) == JointAlphaVector

def is_decisionRule(decision_rule):
    assert type(decision_rule) == DecisionRule

def is_betaVector(beta_vector):
    assert type(beta_vector) == BetaVector

def is_beliefSpace(belief_space):
    assert type(belief_space) == BeliefSpace
    





