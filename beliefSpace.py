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
        - The class keeps track of the mapping between belief states and its IDs.
        - It also maintains a network of belief ids that shows the connections between different belief ids bridged by action,observation pairs.
        - the class also maintains a dictionary of the existing belief ids at a given timestep (time_index_table)
    """
    def __init__(self,horizon,density):
        self.horizon = horizon
        self.id = 1
        self.initial_belief = BeliefState(np.array(PROBLEM.GAME.b0)) 
        self.monte_carlo_samples = 5000

        #set density
        self.density = density

        #initialize belief dictionary that stores the mapping of belief states to belief ids, we reserve id "0" for the initial belief
        self.belief_dictionary = {0:self.initial_belief}

        #initialize time_index_table dictionary that stores belief ids at every timestep as a set (so as to only keep unique ids)
        self.time_index_table = {0: set([0])}
        for timestep in range(1,horizon+1): self.time_index_table[timestep] = set()

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

    def update_time_index_table(self,belief_id,next_belief_id):
        for timestep in range(1,self.horizon-1):
            if belief_id in self.time_index_table[timestep]:
                self.time_index_table[timestep+1].add(next_belief_id)


    

#===== public methods ===============================================================================================

    def size(self):
        "returns number of beliefs in the belief space"
        return self.id
    
    def size_at_horizon(self,horizon):
        unique_belief_ids = set()
        for timestep in range(horizon):
            unique_belief_ids.union(self.time_index_table[timestep])
        return len(unique_belief_ids)
    
    def set_density(self,density):
        "function to externally set the density of the Belief Space object"
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

        # initialize belief dictionary that keeps mapping of beliefs to belief_ids, belief id 0 always belongs to the initial beleif
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
                            original_next_belief = self.get_belief(current_belief_id).next_belief(joint_action,joint_observation)
                        
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
        print(f"\tbelief expansion to horizon {self.horizon} done with density = {self.density} , resulting belief space size = {self.size()}\n")
        return self

    def monte_carlo_expansion(self):
        """samples the belief space by using monte carlo sampling methods by sampling trajectories of games for a fixed horizon.
           belief states that make up a given trajectory will be recorded and added to the bag of beliefs so long as it is "sufficiently different" from other belief states.  
        
        ## pseudo code : 
            for _ in iterations :
                ## to sample a single trajectory :: 
                b = initial_belief
                for t in [1,H-1]:
                    u <= sample(joint_actions)
                    z <= sample(joint_observations)
                        if Pr(z|b,u)>0 :
                            b_t+1 = Transition_fn(b_t,u,z)
                            if sufficiently_different(b_t+1) : add b_t+1 to belief Space and network
                            else : use an existing b_t+1 in the bag of beliefs
                b <= b_t+1

        """
        for iters in range(self.monte_carlo_samples):
            # print(f"{iters}/100")
            current_belief_id = 0
            for timestep in range(0,self.horizon-1):
                joint_action = np.random.choice(PROBLEM.JOINT_ACTIONS)
                joint_observation = np.random.choice(PROBLEM.JOINT_OBSERVATIONS,p=self.get_belief(current_belief_id).coniditioned_observation_distribution(joint_action))
                if utilities.observation_probability(joint_observation,self.get_belief(current_belief_id),joint_action)>0:
                    # calculate next belief using joint_action and joint_observation
                    next_belief = self.belief_dictionary[current_belief_id].next_belief(joint_action,joint_observation)
                
                    # check if the new belief is sufficiently different and add to bag of beliefs
                    next_belief_id = self.new_belief_subroutine(current_belief_id,next_belief,joint_action,joint_observation,timestep)
                                    
                    # update the connection in the time index table                 
                    self.update_connections(current_belief_id,next_belief_id)

                    # update current_belief_id for the next iteration
                    current_belief_id = next_belief_id
        print(f"\tMonte-Carlo belief expansion done with density = {self.density} , resulting belief space size = {self.size()}\n")

    def add_samples(self,num_trajectories,density=None):
        "function that samples more trajectories and newly encountered belief states into the bag of beliefs"

        # if a density value is given, then set the density to the given value 
        if density is not None : self.density = density

        new_points = 0

        for _ in range(num_trajectories):
            current_belief_id = 0
            for timestep in range(0,self.horizon-1):
                current_belief = self.get_belief(current_belief_id)
                joint_action = np.random.choice(PROBLEM.JOINT_ACTIONS)
                joint_observation = np.random.choice(PROBLEM.JOINT_OBSERVATIONS,p=current_belief.coniditioned_observation_distribution(joint_action))
                next_belief = current_belief.next_belief(joint_action,joint_observation)


                # check if the calculated next_belief is sufficiently different and add to bag of beliefs, if not, get the closest belief 
                next_belief_id = self.new_belief_subroutine(current_belief_id,next_belief,joint_action,joint_observation,timestep)
                self.update_connections(current_belief_id,next_belief_id)

                if next_belief_id == next_belief.id : new_points+=1

                current_belief_id = next_belief_id
        print(f"Added samples to the beleif space, with density ={self.density} , new belief space size = {self.size()} with {new_points} new belief points")
        self.print_belief_table()
        return self.size()


    

    def new_belief_subroutine(self,current_belief_id,next_belief,joint_action,joint_observation,timestep):
        """function that defines subroutines that deals with a new belief state
            it first checks if the new belief state is sufficiently different from other belief states in the bag, given the density parameter that is set. 
            if it is, then it is assigned a belief id and added to the network and the bag.
            if it is not, we look for an existing belief in the bag that is closest in distance to the new belief state, and we use that beleif state instead.

        """


        # check if the next belief state's is "sufficiently different"
        if self.distance(next_belief):
            # add next belief to the bag of beliefs
            next_belief_id = self.id 
            self.add_network_connection(current_belief_id,joint_action,joint_observation,next_belief_id)
            self.add_new_belief_in_bag(next_belief,timestep+1)

        # if it is not sufficiently different, we use a stored belief state that is closest in distance to original_next_belief 
        elif len(self.belief_dictionary):
            # if there is no existing connection in the network, we add the new connection with an existing belief that is closest to the calculated belief 
            if self.check_network_connection(current_belief_id,joint_action,joint_observation)==False:
                _,next_belief_id = self.get_closest_belief(next_belief)
                # add closest belief id to belief mapping at current timestep 
                self.time_index_table[timestep+1].add(next_belief_id)
                self.add_network_connection(current_belief_id,joint_action,joint_observation,next_belief_id)

            
            # if there is an existing connection, then we simply use the next belief from the existing network connection 
            else:
                next_belief_id = self.get_network_connection(current_belief_id,joint_action,joint_observation)
                self.time_index_table[timestep+1].add(next_belief_id)


       
        return next_belief_id
    

    def update_connections(self,previous_belief_id,next_belief_id):
        # print("UDPATING")
        for timestep,belief_list in self.time_index_table.items():
            if previous_belief_id in belief_list and timestep+1<self.horizon:
                self.time_index_table[timestep+1].add(next_belief_id)
                # print("connections updated")
