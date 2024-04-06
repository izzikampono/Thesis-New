import numpy as np
from decisionRule import DecisionRule
import sys
from docplex.mp.model import Model   
from problem import PROBLEM
from stackelbergMilp import MILP
PROBLEM = PROBLEM.get_instance()  

""" Module that defines the linear programs and solution methods for the "State of the art" solution of a given game """


def get_joint_DR(leader_decision_rule,follower_decision_rule):
    """function to caculate the joint decision rule from individual decision rules"""
    joint_DR = {}
    for state in PROBLEM.STATES:
        joint_DR[state] = np.zeros((len(PROBLEM.JOINT_ACTIONS)))
        for leader_action,leader_action_probability in enumerate(leader_decision_rule):
            for follower_action,follower_action_probability in enumerate(follower_decision_rule[state]):
                joint_DR[state][PROBLEM.get_joint_action(leader_action,follower_action)] = leader_action_probability * follower_action_probability
    return joint_DR

def cooperative_sota(belief,beta):
    """a state of the art mixed integer linear program to solve for the optimal decision rule for common reward subgames rooted at a belief state b """
    best_leader_action = 0
    best_follower_value = -np.inf 
    best_follower_decision_rule = {}

    # get follower's best response for each leader_action 
    for leader_action in PROBLEM.ACTIONS[0]:
        
        follower_value = 0
        follower_decision_rule = {}
        for state in PROBLEM.STATES:
            max = -np.inf
            for follower_action in PROBLEM.ACTIONS[1]:
                value = beta.two_d_vectors[0][state][PROBLEM.get_joint_action(leader_action,follower_action)]
                if value > max:
                    max = value
                    follower_decision_rule[state] = follower_action
            follower_value += belief.value[state] * max

        #save best joint action combination from followers pov
        if follower_value > best_follower_value:
            best_leader_action = leader_action
            best_follower_value = follower_value
            best_follower_decision_rule = follower_decision_rule
    
    # get binary DR of follower a(u|x)
    follower_DR = {}
    for state in PROBLEM.STATES:
        follower_DR[state] = np.identity(len(PROBLEM.ACTIONS[PROBLEM.LEADER]))[best_follower_decision_rule[state]]
  
    leader_DR = np.identity(len(PROBLEM.ACTIONS[PROBLEM.LEADER]))[best_leader_action]
    joint_DR = get_joint_DR(leader_DR,follower_DR)

    # return leader-follower values and DR
    return best_follower_value, best_follower_value, DecisionRule(leader_DR,follower_DR,joint_DR)

def stackelberg_sota(belief,beta):
    """a state of the art mixed integer linear program to solve for the optimal decision rule for general sum subgames rooted at a belief state b """
    milp = MILP(belief,beta)
    return milp.run()
  
def zerosum_sota(belief,beta):
    """function to solve for optimal DR at a given subgame for zerosum games. This function uses different linear programs for the leader and follower """
    leader_value , leader_DR =  zerosum_lp_leader(belief,beta)
    follower_value , follower_DR =  zerosum_lp_follower(belief,beta)
    return leader_value,follower_value, DecisionRule(leader_DR,follower_DR,get_joint_DR(leader_DR,follower_DR))

def zerosum_lp_leader(belief,beta):
    "state of the art linear program to solve for a subgame of a zerosum game from the perspective of the leader"
    lp = Model(f"{PROBLEM.GAME.name} zerosum leader problem")

    #initialize linear program variables
    DR = lp.continuous_var_list(len(PROBLEM.ACTIONS[0]),name = [f"a0_{i}" for i in PROBLEM.ACTIONS[0]],ub=1,lb=0) # type: ignore
    V = lp.continuous_var_list(len(PROBLEM.STATES) ,name=[f"V_{state}" for state in PROBLEM.STATES],ub=float('inf'),lb=float('-inf')) # type: ignore

    # define objective function 
    obj_fn = 0
    for state,V_state in enumerate(V) : 
        obj_fn += belief.value[state] * V_state
    lp.maximize(obj_fn)

    # define constraints 
    for state,state_value in enumerate(V):
        for opponent_action in PROBLEM.ACTIONS[1]:    
            lhs = 0   
            for leader_action, leader_action_probability in enumerate(DR):
                lhs += beta.two_d_vectors[0][state][PROBLEM.get_joint_action(leader_action,opponent_action)] * leader_action_probability
            lp.add_constraint(lhs>=state_value)

    #add sum-to-one constraint
    lp.add_constraint(lp.sum(DR) == 1)

    #solve and export 
    sol = lp.solve()
    # lp.export_as_lp(f"zerosum_lp_leader")
    return lp.solution.get_objective_value(),lp.solution.get_value_list(DR) # type: ignore

def zerosum_lp_follower(belief,beta):
    "state of the art linear program to solve for a subgame of a zerosum game from the perspective of the follower"
    lp = Model(f"{PROBLEM.GAME.name} problem")

    #initialize linear program variables
    V = lp.continuous_var(name="V",ub=float('inf'),lb=float('-inf'))
    decision_rule = {}
    for state in PROBLEM.STATES:
        decision_rule[state] = lp.continuous_var_list(len(PROBLEM.ACTIONS[PROBLEM.FOLLOWER]), name = [f"a1_u{i}_x{state}" for i in PROBLEM.ACTIONS[1]])  # type: ignore
       
    # define objective function 
    lp.minimize(V)

    # define constraints 
    # constraint for every x and u^2 : \sum_{x} \sum_{u^1} += b(x) a(u^2) beta(b,x,u^1,u^2)  <= V
    for leader_action in PROBLEM.ACTIONS[0]:
        rhs = 0
        for state in PROBLEM.STATES:
            for follower_action, follower_action_probability in enumerate(decision_rule[state]):            
                rhs += belief.value[state] * beta.two_d_vectors[0][state][PROBLEM.get_joint_action(leader_action,follower_action)] * follower_action_probability
        lp.add_constraint(V>=rhs)

    #add sum-to-one constraint for decision rule
    for state in PROBLEM.STATES: lp.add_constraint(lp.sum(decision_rule[state]) == 1)

    #solve and export 
    sol = lp.solve()
    if sol:
        lp.export_as_lp(f"zerosum_lp_follower")
        follower_DR = {}
        for state in PROBLEM.STATES:
            follower_DR[state] = lp.solution.get_value_list(decision_rule[state])  # type: ignore
        return lp.solution.get_objective_value(), follower_DR  # type: ignore
    else: 
        print("CANNOT SOLVE ZEROSUM FOLLOWER LINEAR PROGRAM ")
        sys.exit()
    return



