from xmlrpc.client import Boolean
import numpy as np 
import sys
from traitlets import Bool
from beliefState import BeliefState
from problem import PROBLEM
PROBLEM = PROBLEM.get_instance()
import utilities
from typing import Union
import gc
gc.enable()



class BeliefSpace:
    """Class that represents the Belief Space, a structure that maintains all belief states of the game.
        The class keeps track of the mapping between belief states and its IDs.
        It also maintains a network of belief ids that shows the connections between different belief ids bridged by action,observation pairs

    """
    def __init__(self,horizon,density):
        self.horizon = horizon
        self.id = 1
        self.initial_belief = BeliefState(np.array(PROBLEM.GAME.b0)) 
        #set density
        self.density = density

        #initialize belief dictionary that stores the mapping of belief states to belief ids, we reserve id "0" for the initial belief
        self.belief_dictionary = {0:self.initial_belief}

        #initialize time_index_table dictionary that stores belief ids at every timestep as a set (so as to only keep unique ids)
        self.time_index_table = {0: set([0])}
        for timestep in range(1,horizon+2): self.time_index_table[timestep] = set()

        #initialize network that keeps track of connections between belief ids  
        self.network = {}

#===== private methods ===============================================================================================
    
    def check_network_connection(self,belief_id,joint_action,joint_observation) -> Bool:
        """function to check if a given (action,observation) key already points to an existing next_belief_id in the network"""
        if belief_id in self.network.keys(): 
            if joint_action in self.network[belief_id]:
                if joint_observation in  self.network[belief_id][joint_action] and self.network[belief_id][joint_action][joint_observation] is not None:
                    return True
        return False

    def add_network_connection(self,belief_id,joint_action,joint_observation,next_belief_id):
        """function to add new connection in the belief space network by using an (action,observation) key that points to the resulting next_belief_id"""
        if belief_id not in self.network.keys(): 
            self.network[belief_id]= {}
        if joint_action not in self.network[belief_id].keys():
            self.network[belief_id][joint_action] = {}
        self.network[belief_id][joint_action][joint_observation] = next_belief_id
      
    def get_network_connection(self,belief_id,joint_action,joint_observation):
        return self.network[belief_id][joint_action][joint_observation] 

    def add_new_belief_in_bag(self,belief,timestep):
        """function to store a newly discovered belief by adding it to the diction """
        belief.set_id(self.id)
        self.belief_dictionary[self.id]= belief
        self.time_index_table[timestep].add(self.id)
        self.id += 1

    def distance(self,belief) -> Bool:
        """function to check if a new belief states is "sufficiently different" from other belief states in the bag by calculating its minimum distance from all other belief states and checking if that distance is greater than the density value """
        if len(self.belief_dictionary)<=1: return True
        min_belief = min(self.belief_dictionary.values(), key=lambda stored_belief: np.linalg.norm(stored_belief.value-belief.value))
        min_magnitude = np.linalg.norm(min_belief.value-belief.value)
        return min_magnitude > self.density
    
    def get_closest_belief(self,belief) -> Union[BeliefState,None]:
        """ returns a stored belief and belief_id in the belief space that is closest in distance to the input belief """
        if not self.distance(belief) :

            closest_belief_id, min_belief_value = min(self.belief_dictionary.items(), key=lambda stored_belief: np.linalg.norm(stored_belief[1].value - belief.value))
            if closest_belief_id!=None: return self.get_belief(closest_belief_id),closest_belief_id
            else : 
                print("err0r : no belief found")
                print(min_belief_value)
                print(closest_belief_id)
                sys.exit()
        else : 
            print("New belief encountered, improper sampling!")
            sys.exit()
    

#===== public methods ===============================================================================================

    def size(self):
        "returns number of beliefs in the belief space"
        return self.id+1
    
    def set_density(self,density):
        self.density = density
    
    def get_belief(self,belief_id) -> BeliefState:
        """function to retrieve a belief state that corresponds to a certain belief ID"""
        return self.belief_dictionary[belief_id]

    def reset(self,density=None):
        """function to reset the belief space for new games"""
        # reset id counter
        self.id = 1

        # change density value if optional argument is provided
        if density is not None : self.set_density(density)

        # initialize belief dictionary that keeps mapping of beliefs to belief_ids
        self.belief_dictionary = {0:self.initial_belief}

        # initialize time_index_table dictionary that stores beliefs at every timestep as a set (so as to only keep unique ids)
        self.time_index_table = {0:set([0])}
        for timestep in range(1,self.horizon+1):
            self.time_index_table[timestep] = set()
        
        # reset network
        self.network = {}
        return self
    
    def existing_next_belief_id(self,current_belief_id,joint_action,joint_observation) -> Union[int,None]:
        """function to retrieve an existing network connection from belief_id to next_belief_id using an action,observation key"""
        # check if there is a connection with the (u,z) branch and check if Pr(z|b,u)>0
        if self.check_network_connection(current_belief_id,joint_action,joint_observation) and utilities.observation_probability(joint_observation, self.get_belief(current_belief_id),joint_action)>0:
           return self.network[current_belief_id][joint_action][joint_observation]
         
        return None
    
    def print_belief_table(self):
        for timestep in range(self.horizon):
            print(f"{timestep} : {self.time_index_table[timestep]} ")
    
    def print_network(self):
        """function to print the network that shows connections between belief ids"""
        for belief_id in self.network.keys():
            print(f"  ∟ belief {belief_id} : {self.get_belief(belief_id).value}")
            for joint_action in self.network[belief_id]:
                for joint_observation in self.network[belief_id][joint_action]:
                    if self.network[belief_id][joint_action][joint_observation] is not None:
                        print(f"      ∟ action {joint_action}, observation {joint_observation} : belief {self.network[belief_id][joint_action][joint_observation]}")

    def expansion(self):
        """expands  belief space tree to all branches within the given horizon and stores all new belief states so long as it satisfies the density sufficiency
        
        ## pseudo code : 
            for t in horizon:
                for b_t in belief_table[t]::
                    for u in U_joint :
                        for z in Z_joint :
                            if Pr(z|b,u_j)>0 :
                                b_t+1 = Transition_fn(b_t,u,z)
                                if sufficiently_different(b_t+1) : add to b_t+1 to belief Space and network

        """
        
        
        for timestep in range(0,self.horizon-1):
            for current_belief_id in self.time_index_table[timestep]:
                for joint_action in PROBLEM.JOINT_ACTIONS:
                    for joint_observation in PROBLEM.JOINT_OBSERVATIONS:
                        if utilities.observation_probability(joint_observation,self.get_belief(current_belief_id),joint_action)>0:
                            # calculate next belief using joint_action and joint_observation
                            original_next_belief = self.belief_dictionary[current_belief_id].next_belief(joint_action,joint_observation)
                        
                            # check if the next belief state's is "sufficiently different"
                            if self.distance(original_next_belief):
                                # add belief to the belief network and belief bag
                                self.add_network_connection(current_belief_id,joint_action,joint_observation,self.id)
                                self.add_new_belief_in_bag(original_next_belief,timestep+1)

                            # if it is not sufficiently different, we use a stored belief state that is closest in distance to original_next_belief 
                            elif len(self.belief_dictionary):
                                # if there is no existing connection, we add the new connection with the existing belief
                                if self.check_network_connection(current_belief_id,joint_action,joint_observation)==False:
                                    _,existing_belief_id = self.get_closest_belief(original_next_belief)
                                    self.add_network_connection(current_belief_id,joint_action,joint_observation,existing_belief_id)
                                else:
                                    existing_belief_id = self.get_network_connection(current_belief_id,joint_action,joint_observation)
                                    self.add_network_connection(current_belief_id,joint_action,joint_observation,existing_belief_id)

                                # add closest belief id to belief mapping at current timestep 
                                self.time_index_table[timestep+1].add(existing_belief_id)
        print(f"\tbelief expansion done with density = {self.density} , resulting belief space size = {self.size()}\n")

    