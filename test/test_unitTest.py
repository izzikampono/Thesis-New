
from alphaVector import AlphaVector
from beliefState import BeliefState
from betaVector import BetaVector
from jointAlphaVector import JointAlphaVector


def is_beliefState(belief_state):
    assert type(belief_state) == BeliefState

def is_alphaVector(alpha):
    assert type(alpha) == AlphaVector

def is_betaVector(beta):
    assert type(beta) == BetaVector

def is_jointAlphaVector(alpha):
    assert type(alpha) == JointAlphaVector





