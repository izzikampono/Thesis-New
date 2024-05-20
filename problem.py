from __future__ import barry_as_FLUFL
import numpy as np
from decpomdp import DecPOMDP
import random


__all__ = ["PROBLEM"]
__author__ = "Isabelle Kampono"
__version__ = "0.1.0"

import sys

class PROBLEM:
    """Instantiable Class that holds constant values for a given Problem"""
    _instance = None
    def __init__(self,problem) :
        self.HORIZON = problem.horizon
        self.NAME = problem.name
        self.GAME = problem
        self.STATES = [i for i in range(len(problem.states))]
        self.ACTIONS = [[i for i in range(len(problem.actions[0]))],[j for j in range(len(problem.actions[1]))]]
        self.JOINT_ACTIONS = [i for i in range(len(problem.joint_actions))]
        self.JOINT_OBSERVATIONS = [i for i in range(len(problem.joint_observations))]
        self.TRANSITION_FUNCTION = problem.transition_fn
        self.OBSERVATION_FUNCTION = problem.observation_fn
        # self.REWARDS = self.initialize_rewards()
        self.GAME.reset()
        self.initialize_rewards()
        self.action_dictionary={}
        n=0
        for leader_action in self.ACTIONS[0]:
            for follower_action in self.ACTIONS[1]:
                self.action_dictionary[f"{leader_action}{follower_action}"] = n
                n+=1

        self.action_dict = dict((val, key) for key, val in self.action_dictionary.items())
        self.LEADER = 0
        self.FOLLOWER = 1

#===== private methods ===============================================================================================
    def follower_stackelberg_reward(self):
        """ function that initializes the followers reward in a general sum game """
        #fix random seed so that generated rewards stay the same at different iterations 
        seed_value = 42
        random.seed(seed_value)

        #initialize reward and upper/lower bounds for opponenet reward
        stackelberg_follower_reward = np.zeros(self.GAME.reward_fn_sa.shape)
        min_NUM = int(min([min(row)for row in self.GAME.reward_fn_sa]))
        max_NUM = int(max([max(row) for row in self.GAME.reward_fn_sa]))

        # Generate reward within the upper/lower bound
        for joint_action in self.JOINT_ACTIONS:
            for state in self.STATES:
                stackelberg_follower_reward[joint_action][state]+=random.randint(min_NUM, max_NUM)
        return stackelberg_follower_reward

    def dectiger_reward(self):

        return np.zeros(np.size(self.GAME.reward_fn_sa))

    
    def initialize_rewards(self):
        #Competitive reward matrix indexed by joint actions
        self.REWARDS = { "cooperative" : [self.GAME.reward_fn_sa,self.GAME.reward_fn_sa],
                    "zerosum" : [self.GAME.reward_fn_sa,self.GAME.reward_fn_sa*-1],
                    "generalsum" :[self.GAME.reward_fn_sa,self.follower_stackelberg_reward()]
                    }
        
        # custom made general sum game rewards :: 

        if self.NAME == "dectiger":
            self.REWARDS["generalsum"] = [
                    # leader general sum reward
                    np.array([[-2., -2.],
                            [-40., 50.],
                            [-4., -4.],
                            [-100., 60.],
                            [-50., 70.],
                            [-100., 60.],
                            [25., -20.],
                            [-15., 30.],
                            [30., -20.]]),
                    # follower general sum reward :
                    np.array([[-2., -2.],
                            [-20., 25.],
                            [60., -100.],
                            [-4., -4.],
                            [-20., 30.],
                            [60., -100.],
                            [50., -40.],
                            [30., -15.],
                            [70., -50.]]),
            ]
        if self.NAME == "broadcastChannel":
            self.REWARDS["generalsum"] = [
                                    # leader reward 
                                    np.array([[-1,-1,-1,-2],
                                    [0,0,1,1],
                                    [-1,0,-1,0],
                                    [0,0,0,0]
                                    ]),

                                    #follower reward
                                    np.array([[-1,-1,-1,-2],
                                    [-1,-1,0, 0,],
                                    [0,1,0,1],
                                    [0,0,0,0]
                                    ])
                                    ]
        if self.NAME=="2generals":
            self.REWARDS["generalsum"] = [np.array([[-2,-1],
                                        [-10,-10],
                                        [-11,-11],
                                        [5,-15]
                                        ]),
                                        np.array([[-1,-1],
                                        [-10,-10],
                                        [0,0],
                                        [15,-5]
                                        ])]
  
        

      
                                  
            


#===== public methods ===============================================================================================
    @classmethod
    def get_instance(cls):
        """function that returns the initialized instance of the class  """

        if cls._instance is None:
            raise ValueError("Constants has not been initialized.")
        return cls._instance

    @classmethod
    def initialize(cls, value):
        """function to initialize the only singular instance of this class """
        if cls._instance is not None:
            raise ValueError("Constants has already been initialized.")
        cls._instance = cls(value)

    def get_joint_action(self,leader_action,follower_action):
        "returns the joint action of two seperate leader and follower actions "
        return self.action_dictionary[f"{leader_action}{follower_action}"]
    
    def get_seperate_action(self,uj):
        """function to seperate a joint action into leader action and follower action """
        action_Join = int(self.action_dict[uj])
        return action_Join//10,action_Join%10