import numpy as np
from decisionRule import DecisionRule
from problem import PROBLEM
import sys
import copy
from docplex.mp.model import Model
PROBLEM = PROBLEM.get_instance()
import gc 
gc.enable()
class LP:
    """Class that represents the linear program to generate possible reward for cybersecurity games """
    def __init__(self,reward_fn) :
        self.lp = None
        self.individual_reward_fn = reward_fn

    def initialize_central_planner_reward(self):
        matrix = np.zeros(PROBLEM.STATES,PROBLEM.JOINT_ACTIONS)

    def get_bounds(self,state,joint_action):
        """function that returns the upper bound and lower bound of a specific R(x,u1,u2) 
           returns :: Tuple(upper_bound,lower_bound)
        """
        return np.max(self.individual_reward_fn[state][joint_action]),np.min(self.individual_reward_fn[state][joint_action])

    def solve(self):

        self.lp = Model(f"{PROBLEM.GAME.name} central planner reward matrix")

        # Define model variables :: Central planner reward function
        matrix = {}
        for state in PROBLEM.STATES:
            matrix[state] = {}
            for joint_action in PROBLEM.JOINT_ACTIONS:
                ## define each R_j(x,u1,u2)
                follower_action,leader_action = PROBLEM.get_seperate_action(joint_action)

                ## get bounds
                ub,lb = self.get_bounds(state,joint_action)
                matrix[state][(follower_action,leader_action)] = self.lp.integer_var(name=f"R({state},{follower_action,leader_action})",ub = ub,lb=lb)



        # Define constraint from follower perspective ::

        # FIX STATE 
        for state in PROBLEM.STATES :
            # fix leader action
            for leader_action in PROBLEM.ACTIONS[PROBLEM.LEADER]:
                # fix the follower action
                for follower_action in PROBLEM.ACTIONS[PROBLEM.FOLLOWER]:
                    follower_actions = copy.copy(PROBLEM.ACTIONS[PROBLEM.FOLLOWER])
                    follower_actions.remove(leader_action)
                    # Get other follower actions that is not equal to the fixed follower action
                    for follower_action_ in follower_actions:
                        joint_action_1 = PROBLEM.get_joint_action(leader_action,follower_action)
                        joint_action_2 = PROBLEM.get_joint_action(leader_action,follower_action_)
                        # add constraint :: R_j(x,u1,u2) -  R_j(x,u1,u2') = R_follower(x,u1,u2) - R_follower(x,u1,u2')
                        self.lp.add_constraint(matrix[state][(leader_action,follower_action)] - matrix[state][(leader_action,follower_action_)] == self.individual_reward_fn[state][joint_action_1][PROBLEM.FOLLOWER] - self.individual_reward_fn[state][joint_action_2][PROBLEM.FOLLOWER])
                    
                        ## add cannot equal to 0 constraint
                        self.lp.add_constraint(matrix[state][(leader_action,follower_action)] - matrix[state][(leader_action,follower_action_)] != 0 )

        # Define constraint from leader perspective ::

        # FIX STATE 
        for state in PROBLEM.STATES :
            # fix follower action
            for follower_action in PROBLEM.ACTIONS[PROBLEM.FOLLOWER]:
                # fix the leader action
                for leader_action in PROBLEM.ACTIONS[PROBLEM.LEADER]:
                    leader_actions = copy.copy(PROBLEM.ACTIONS[PROBLEM.FOLLOWER])
                    # Get other leader actions that is not equal to the fixed leader action
                    leader_actions.remove(leader_action)
                    for leader_action_ in leader_actions:
                        joint_action_1 = PROBLEM.get_joint_action(leader_action,follower_action)
                        joint_action_2 = PROBLEM.get_joint_action(leader_action_,follower_action)
                        # add constraint :: R_j(x,u1,u2) -  R_j(x,u1',u2) = R_leader(x,u1,u2) - R_leader(x,u1',u2)
                        self.lp.add_constraint(matrix[state][(leader_action,follower_action)] - matrix[state][(leader_action_,follower_action)] == self.individual_reward_fn[state][joint_action_1][PROBLEM.LEADER] - self.individual_reward_fn[state][joint_action_2][PROBLEM.LEADER])
                        
                        # add cannot be equal to zero (!= 0) contraint
                        self.lp.add_constraint(matrix[state][(leader_action,follower_action)] - matrix[state][(leader_action_,follower_action)] != 0 )

        ## try solving the model
        solution = self.lp.solve()

        # Check if a solution was found and print results
        if solution:
            print(f'Solution status: {solution.solve_status}')
            for state in matrix:
                for action in matrix[state]:
                    print(f'{state}, {action} = {solution.get_value(matrix[state][action])}')
        else:
            print('No solution found')
            # print("Infeasible constraints:")
            # for constraint in self.lp.iter_constraints():
            #     expr = constraint.expression
                


            print(f"\n\n LP exported as {PROBLEM.NAME}_Reward_LP")
            return


