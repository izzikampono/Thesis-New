import numpy as np 

class DecisionRule:
    """the DecisionRule class holds probability distributions over actions for both the leader and the follower.
    It also maintains the joint decision rule which is a probability distribution over joint actions 
    """
    
    def __init__(self,a0,a1,aj):
        self.leader = a0
        self.follower = a1
        self.joint = aj
        pass