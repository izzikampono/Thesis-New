import numpy as np
from decisionRule import DecisionRule
from problem import PROBLEM
import sys
from docplex.mp.model import Model
PROBLEM = PROBLEM.get_instance()
import gc 
gc.enable()
class MILP:
    """Class that represents the mixed integer linear program to use when game is solved using the Stackelberg method """
    def __init__(self,belief,beta) :
        self.milp = None
        self.belief = belief
        self.beta = beta
        pass

#===== private methods ===============================================================================================


    def set_objective_function(self,leader_decision_rule,follower_decision_rule):
        """function that sets the objective function to be maximized by the milp
           objective function :: maximize V^1(b,a^1,a^2) = \sum{x} \sum{u^joint} b(x) * a^1(u) * a^2(u|x) *  beta(x,u^joint)
        """
        objective_fn = 0
        for state in PROBLEM.STATES:
            for joint_action in PROBLEM.JOINT_ACTIONS:
                    leader_action, follower_action = PROBLEM.get_seperate_action(joint_action)
                    objective_fn += self.belief.value[state] * self.beta.two_d_vectors[PROBLEM.LEADER][state][joint_action]  * leader_decision_rule[leader_action] * follower_decision_rule[state][follower_action]
        self.milp.maximize(objective_fn)
    
    def set_sum_to_one_constraints(self,leader_DR,follower_DR,joint_DR):
        self.milp.add_constraint(self.milp.sum(leader_DR)==1)
        for state in PROBLEM.STATES:
            self.milp.add_constraint(self.milp.sum(follower_DR[state])==1)
            self.milp.add_constraint(self.milp.sum(joint_DR[state])==1)

    def set_seperability_constraints(self,leader_decision_rule,follower_decision_rule,joint_decision_rule):
        """function that adds seperaibility constrains to the MILP to ensure that the joint decision rule can be seperated into individual decision rules for each player"""
        
        # First, add seperability constraint from leaders point of view by fixing the leaders action and summing over follower action probabilities
        #fix state
        for state in PROBLEM.STATES:
            # fix leader action and sum over follower actions 
            for leader_action, leader_action_probability in enumerate(leader_decision_rule):
                sum = 0
                for follower_action,follower_action_probability in enumerate(follower_decision_rule[state]):
                    sum += joint_decision_rule[state][PROBLEM.get_joint_action(leader_action,follower_action)]
                #add constraint for each state and leader action :: \sum_{u_follower} joint_DR(u_leader,u_follower) == leader_DR(u_leader)
                self.milp.add_constraint(sum == leader_action_probability)

        # Then, add seperability constraint from followers point of view by fixing follower action and summing over the leaders action probability
        for state in PROBLEM.STATES:
            # fix follower action and sum over leader actions
            for follower_action,follower_action_probability in enumerate(follower_decision_rule[state]):
                sum = 0
                for leader_action, leader_action_probability in enumerate(leader_decision_rule):
                    # get sum of joint DR with a fixed leader_action 
                    sum += joint_decision_rule[state][PROBLEM.get_joint_action(leader_action,follower_action)]
                #add constraint for each state and leader action :: \sum_{x} \sum_{u_leader} joint_DR(u_leader,u_follower|x) == follower_DR(u_follower|x)
                self.milp.add_constraint(sum == follower_action_probability)

    def calculate_follower_DR(self,leader_decision_rule,state)-> list:
        max_value = -np.inf
        best_follower_action = None
        for follower_action in PROBLEM.ACTIONS[PROBLEM.FOLLOWER]:
            value = 0
            for leader_action in PROBLEM.ACTIONS[PROBLEM.LEADER]:
                value += leader_decision_rule[leader_action] * self.beta.two_d_vectors[PROBLEM.FOLLOWER][state][PROBLEM.get_joint_action(leader_action,follower_action)]
            if value > max_value:
                max_value = value
                best_follower_action = follower_action
        # turn deterministic action to a decision rule 
        follower_decision_rule = np.identity(len(PROBLEM.ACTIONS[PROBLEM.FOLLOWER]))[best_follower_action]
        return (follower_decision_rule,max_value)
    
    def calculate_follower_DR2(self,leader_decision_rule,state)-> list:
        milp = Model("follower_action")
        follower_DR = {}
        for state in PROBLEM.STATES:
            follower_DR[state] = self.milp.binary_var_list(len(PROBLEM.ACTIONS[PROBLEM.FOLLOWER]), name = [f"a1_u{i}_x{state}" for i in PROBLEM.ACTIONS[PROBLEM.LEADER]]) # type: ignore
        
        for follower_action in PROBLEM.ACTIONS[PROBLEM.FOLLOWER]:
            value = 0
            for leader_action in PROBLEM.ACTIONS[PROBLEM.LEADER]:
                value += leader_decision_rule[leader_action] * self.beta.two_d_vectors[PROBLEM.FOLLOWER][state][PROBLEM.get_joint_action(leader_action,follower_action)]
            if value > max_value:
                max_value = value
                best_follower_action = follower_action
        # turn deterministic action to a decision rule 
        follower_decision_rule = np.identity(len(PROBLEM.ACTIONS[PROBLEM.FOLLOWER]))[best_follower_action]
        return (follower_decision_rule,max_value)
                




    def get_solution(self,leader_DR,follower_DR,joint_DR):
        """function to solve the mixed integer linear program""" 
        sol = self.milp.solve()
        self.milp.export_as_lp(f"Stackelberg_LP")

        if sol:
            optimal_leader_DR = self.milp.solution.get_value_list(leader_DR) # type: ignore
            optimal_follower_DR = {}
            optimal_follower_values = {}

            # get values of optimal joint DR and optimal individual DRs from solution docplex object
            for state in PROBLEM.STATES:
                optimal_follower_DR[state], optimal_follower_values[state] = self.calculate_follower_DR(optimal_leader_DR,state)
               
            return self.milp.solution.get_objective_value(),optimal_follower_values,DecisionRule(optimal_leader_DR,optimal_follower_DR,None) # type: ignore
        else:
            print("\n\n Err0r : Mixed integer LInear program cannot be solved\n\n")
            sys.exit()
    


#===== public methods ===============================================================================================

    def run(self):
        """Mixed integer linear program to solve for the optimal decision rule of a subgame rooted at a certain belief.
           joint and follower decision rules come in the form of Decision_rule(action|state)
           the function will return : leader value, follower value, and the  decision rules. """
        self.milp = Model(f"{PROBLEM.GAME.name} problem")

        ## initalize MILP variables 
        # leader_DR = a(u) for all u in U[LEADER]
        # follower_DR = a(u|x) for all u in U[FOLLOWER] and for all x in X
        # joint_DR = a(u|x) for all joint_u in U^joint and for all x in X
        leader_DR = self.milp.continuous_var_list(len(PROBLEM.ACTIONS[PROBLEM.LEADER]), name=[f"a0_u{i}" for i in PROBLEM.ACTIONS[PROBLEM.LEADER]],ub=1,lb=0) # type: ignore
        follower_DR = {}
        joint_DR = {}
        for state in PROBLEM.STATES:
            follower_DR[state] = self.milp.binary_var_list(len(PROBLEM.ACTIONS[PROBLEM.FOLLOWER]), name = [f"a1_u{i}_x{state}" for i in PROBLEM.ACTIONS[PROBLEM.LEADER]]) # type: ignore
            joint_DR[state] = self.milp.continuous_var_list(len(PROBLEM.JOINT_ACTIONS),name  = [f"aj_u{i}_x{state}" for i in PROBLEM.JOINT_ACTIONS],ub=1,lb=0)# type: ignore

        # define the objective function to maximize 
        self.set_objective_function(leader_DR,follower_DR)

        ## Add Constraints (subject to) :: V^2(x,a1,a2) < V^2(x,a1,u2) for all u in U2 and x in X
        # first, define lhs of linear equivalence expression equal to V^2(x,a1,a2) - without the belief value 
        lhs = {}
        for state in np.nonzero(self.belief.value)[0]:
            lhs[state] = 0
            for follower_action,follower_action_probability in enumerate(follower_DR[state]):
                for leader_action, leader_action_probability in enumerate(leader_DR):
                    joint_action = PROBLEM.get_joint_action(leader_action,follower_action)
                    #  define V^2(x,a1,a2) :: \sum_{x} \sum_{u_joint} += beta[FOLLOWER](x,u_joint) * joint_DR(u_joint|x)
                    lhs[state] += self.beta.two_d_vectors[PROBLEM.FOLLOWER][state][joint_action]  * joint_DR[state][joint_action] 

        # define right hand side of the linear equivalence expression equal to  V^2(x,a1,u2) for every u in U2 and every x in X using loops
        for state in np.nonzero(self.belief.value)[0]:
            for follower_action,follower_action_probability in enumerate(follower_DR[state]):
                rhs = 0
                #  define V^2(x,a1,u2) for every follower action :: \sum_{x} \sum_{u_leader} += beta[FOLLOWER](x,u_joint) * leader_DR(u_leader)
                for leader_action, leader_action_probability in enumerate(leader_DR):
                    rhs += self.beta.two_d_vectors[PROBLEM.FOLLOWER][state][PROBLEM.get_joint_action(leader_action,follower_action)] * leader_action_probability
                # for every state and follower action, we add the constraint V^2(x,a1,a2) >= V^2(x,a1,u2) to the milp, note that the LHS of the expression remains the same 
                self.milp.add_constraint(lhs[state]>=rhs)

        # add sum to 1 constraint for all Decision Rules
        self.set_sum_to_one_constraints(leader_DR,follower_DR,joint_DR)

        # add seperability constraints for joint_DRs and singular_DRs    
        self.set_seperability_constraints(leader_DR,follower_DR,joint_DR)
        
        return self.get_solution(leader_DR,follower_DR,joint_DR)
    

 