from pickle import FALSE
import string
import numpy as np
import pandas as pd
from traitlets import Bool
from problem import PROBLEM
import time 
from pbvi import PBVI
import matplotlib.pyplot as plt
from beliefSpace import BeliefSpace
from utilities import *
PROBLEM = PROBLEM.get_instance()
import gc
import copy
gc.enable()

class Experiment():
    def __init__(self,horizon,iterations) -> None:
        self.planning_horizon = horizon
        self.iterations = iterations
        self.game = None
        self.policies = {"cooperative" : {},"zerosum" : {},"generalsum" : {}}
        self.database = self.initialize_database()

#===== private methods ===============================================================================================


    def import_experiment_data(self,filename,horizon,iter):
        self.database = pd.read_csv(filename)
        self.planning_horizon = horizon
        self.iterations = iter
   
    def initialize_game(self,horizon,gametype,belief_space):
        """initializes game with horizon and density values for the belief space"""
        self.game = PBVI(horizon=horizon,gametype=gametype,belief_space=belief_space)


    
    def initialize_database(self):
        """function to initiialize database that stores results of each solve of a game.
            the database records several measurements that come with a single solve of a game, i.e : gametype, SOTA (true/false),
            horizon,iterations,time , number of beleif states in the belief space, the value at initial belief, and density
        """
        database = {"horizon": [],
                    "gametype":[],
                    "SOTA" : [],
                            "iterations" : [],
                            "time" : [],
                            "number_of_beliefs" : [],
                            "leader values":[],
                            "follower values":[],
                            "density" : []
                        }
        return database
    
    def add_to_database(self,game_type,horizon,SOTA,num_iterations,average_time,num_beliefs,leader_values,follower_value,density):
        """function to add the result of a solve into the database"""
        sota = {True:"State of the Art" , False:"Stackelberg"}
        self.database["horizon"].append(horizon)
        self.database["gametype"].append(game_type)
        self.database["SOTA"].append(sota[SOTA])
        self.database["iterations"].append(num_iterations)
        self.database["time"].append(average_time)
        self.database["number_of_beliefs"].append(num_beliefs)
        self.database["leader values"].append(leader_values)
        self.database["follower values"].append(follower_value)
        self.database["density"].append(density)
        return
    
    def export_database(self,experiment_type="normal",current_horizon=None):
        """function to save the results of the experiment as a csv file"""
        if experiment_type=="normal":
            if current_horizon!=None:pd.DataFrame(self.database).to_csv(f"raw_results/{PROBLEM.GAME.name} ({current_horizon}).csv", index=False)
            else : pd.DataFrame(self.database).to_csv(f"raw_results/{PROBLEM.GAME.name} ({self.planning_horizon}).csv", index=False)
        else : pd.DataFrame(self.database).to_csv(f"density_experiment_raw_results/{PROBLEM.GAME.name} ({self.planning_horizon}).csv", index=False)

    def construct_stackelberg_comparison_matrix(self,horizon):
        """function to create the general-sum game comparison matrix that shows the performance of the SOTA trained agents against the Stackelberg trained agents """
        print("Constructing Stackelberg Comparison matrix..")
        start_time = time.time()

        # initialize stackelberg comparison matrix 
        matrix = {"Strong Leader Policy" : {"Strong Follower Policy" : None , "Blind Follower Policy" : None},"Weak Leader Policy" : {"Strong Follower Policy" : None, "Blind Follower Policy" : None }}
        
        # populate with values of evaluated policies
        matrix["Strong Leader Policy"]["Strong Follower Policy"] =  self.policies["generalsum"][horizon][False].get_alpha_pairs(0,0).get_value(self.game.value_function.belief_space.initial_belief)
        matrix["Strong Leader Policy"]["Blind Follower Policy"] = self.game.evaluate(0,0,self.policies["generalsum"][horizon][False],self.policies["generalsum"][horizon][True])
        matrix["Weak Leader Policy"]["Blind Follower Policy"] = self.policies["generalsum"][horizon][True].get_alpha_pairs(0,0).get_value(self.game.value_function.belief_space.initial_belief)
        matrix["Weak Leader Policy"]["Strong Follower Policy"] = self.game.evaluate(0,0,self.policies["generalsum"][horizon][True],self.policies["generalsum"][horizon][False])
        print(f"Done.. in {time.time()-start_time} seconds... exporting to csv..")
        
        # convert to dataframe and export to csv file
        self.comparison_matrix = pd.DataFrame(matrix)
        self.comparison_matrix.to_csv(f"comparison_matrix/{PROBLEM.NAME}({horizon}).csv", index=True)        


#===== public methods ===============================================================================================


    def run_single_experiment(self,horizon,belief_space,gametype,density=0.0001):
        "function to run a single experiemnt with one type of benchmark (problem,gametype) for a given number of iterations "
        
        # initialize list to hold values at initial belief and time for each iteration
        leader_values = []
        follower_values = []
        times = []
        self.initialize_game(horizon,gametype,belief_space)


        #initialize game with gametype and sota 
      
        for iter in range(1,self.iterations+1):
            self.game.reset_value_function(self.game.value_function.belief_space)

            for sota in [True,False]:
                print(f"\t\t\t Solving {gametype} {PROBLEM.NAME} GAME Horizon {horizon} WITH SOTA = {self.game.value_function.sota}  ")
                print(f"iteration : {iter}")
                value,time = self.game.solve_game(sota) 
                leader_values.append(value[PROBLEM.LEADER])
                follower_values.append(value[PROBLEM.FOLLOWER])
                times.append(time)

                # add results to database
                self.add_to_database(gametype,horizon,sota,iter+1,time,self.game.value_function.belief_space.size(),value[0],value[1],density)
                self.export_database(experiment_type="densities")

                #add samples to the belief space for the next iteration
                self.game.value_function.belief_space.add_samples(20)
        
                # store policy from last iteration
                if iter == self.iterations-1 : self.policies[gametype][horizon][sota] = copy.deepcopy(self.game.value_function)

        # extract policy from the last iteration
        return leader_values,follower_values,times
    
    def run_single_experiment_set_densities(self,horizon,belief_space,gametype:string,densities : list):
        """function to run a single experiemnt with one type of benchmark (problem,gametype) for a given number of iterations.
            at each iteration, the algorithm chooses a corresponding value from the densities argument. 
        """
        
        # initialize list to hold values at initial belief and time for each iteration
        
        times = []
        self.initialize_game(horizon,gametype,belief_space)

        #initialize game with gametype and sota 
       
        for iter in range(0,self.iterations):
            # add samples to the belief space for the next iteration
            if iter>0: self.game.value_function.belief_space.add_samples(30,densities[iter])

            # reset value function at each iteration, but make sure to use the same belief space 
            self.game.reset_value_function(self.game.value_function.belief_space)
            for sota in [False,True]:
                print(f"\t\t\t Solving {gametype} {PROBLEM.NAME} GAME Horizon {self.game.value_function.horizon} WITH SOTA = {self.game.value_function.sota} , Belief space size = {self.game.value_function.belief_space.size()}")
                print(f"iteration : {iter}")

                #solve game 
                value ,time = self.game.solve_game(sota) #type:ignore
        
                times.append(time)

                # add results to database
                self.add_to_database(gametype,horizon,sota,iter+1,time,self.game.value_function.belief_space.size(),value[0],value[1],densities[iter-1])
                self.export_database(experiment_type="densities")

              

                # store policy from last iteration
                if iter ==self.iterations-1 : self.policies[gametype][horizon][sota] = copy.deepcopy(self.game.value_function)
            
            
        
        return
    
    
    def run_experiments(self,density=0.001):
        """run experiments for all benchmarks of a fixed problem (solves for all gametypes and SOTA mode).
           function version that exports the progress at each horizon 
        """
        original_belief_space = BeliefSpace(self.planning_horizon,density).expansion()
        for horizon in range(1,self.planning_horizon+1):
            for gametype in ["cooperative","zerosum","generalsum"]:
                self.policies[gametype][horizon] = {}
            
                # run game with set number of iterations
                self.run_single_experiment(horizon,copy.deepcopy(original_belief_space),gametype,density)

                # save the value functions of each game  (will be used to evaluate the policy or get the value of the game )
            # construct stackelberg comparison matrix and export 
            self.construct_stackelberg_comparison_matrix(horizon)
        # save databse as a csv file
        self.export_database(current_horizon=horizon)
        self.database = pd.DataFrame(self.database)
        return self.database
    
    def run_experiments_decreasing_density(self,starting_density):
        """run experiments for all benchmarks of a fixed problem with decreasing densities at each iterations (all gametypes and sota mode)"""
        
        # initialize densities and belief space to be used 
        densities = exponential_decrease(starting_density,self.iterations)
        original_belief_space = BeliefSpace(self.planning_horizon,densities[0])

        original_belief_space.monte_carlo_expansion()

        for horizon in range(1,self.planning_horizon+1):
            for gametype in ["cooperative","zerosum","generalsum"]:
                self.policies[gametype][horizon] = {}
                
                # reset the belief space before running different benchmarks 
                self.run_single_experiment_set_densities(horizon,copy.deepcopy(original_belief_space),gametype,densities)
            
            self.density_plot(horizon)
            # construct stackelberg comparison matrix and export 
            self.construct_stackelberg_comparison_matrix(horizon)

        # save databse as a csv file
        self.export_database(current_horizon=horizon)
        self.database = pd.DataFrame(self.database)
        return self.database
    
    def run_experiments_without_comparison(self,density=0.00001):
        """run experiments for all benchmarks of a fixed problem (solves for all gametypes and SOTA mode) without consructing the stackelberg comparison matrix """
        
        for gametype in ["cooperative","zerosum","generalsum"]:
            for sota in [False,True]:
                for horizon in range(1,self.planning_horizon+1):
                    # run game with set number of iterations 
                    self.run_single_experiment(horizon,gametype,sota,density)

                # save the alpha vector at the initial belief state (will be used to get the policy or get the value of the game )
                self.policies[gametype][sota] = self.game.value_function.get_alpha_pairs(0,0)
        
        # save databse as a csv file
        self.export_database()
        self.database = pd.DataFrame(self.database)
        return self.database
        
       
    
    def generate_comparison_tables(self):
        """function that generates a concise table summarizing the results of all gametypes.
            The table highlights the difference of solving each gametype using 2 different solve methods : stackelberg or state of the art
        """
         
        gametypes = ["cooperative","zerosum","generalsum"]
        # create columns for each gametype with subcolumns : 
        columns = pd.MultiIndex.from_product([gametypes, ["State of the Art Leader Value", 'Stackelberg Leader Value']])
        dataframe = pd.DataFrame(columns=columns)

        game_data = []
        for horizon in range(self.planning_horizon):
            new_row_data = []
            for gametype in ["cooperative","zerosum","generalsum"]:
                # get the data of the current benchmark
                current_data = self.database[(self.database["SOTA"]=="Stackelberg")&(self.database["horizon"]==horizon+1)&(self.database["gametype"]==gametype)]
                stackelberg_value = current_data["leader values"].values[0]
                current_data = self.database[(self.database["SOTA"]=="State of the Art")&(self.database["horizon"]==horizon+1)&(self.database["gametype"]==gametype)]
                SOTA_value = current_data["leader values"].values[0]
                # aggregate data together
                new_row_data = new_row_data + [stackelberg_value,SOTA_value]
            game_data.append(new_row_data)
            # populate dataframe with data 
        dataframe = dataframe.merge(pd.DataFrame(game_data, columns=columns), how='outer')
        #export dataframe to csv file
        dataframe.to_csv(f"comparison_table/{PROBLEM.NAME}_{self.planning_horizon}.csv",index=True)
        return dataframe        


    def generate_summary_table(self,densities = False):
        """function that generates a concise table summarizing the results of the experiments.
            uses data stored in self.database 
        """
        algorithms = ['State of the Art','Stackelberg']
        # create columns for each algorithm
        columns = pd.MultiIndex.from_product([algorithms, ['time', 'leader value', 'iteration',"number_of_beliefs"]])
       
       #initialize tables 
        tables = dict.fromkeys(["cooperative","zerosum","generalsum"])

        for gametype in ["cooperative","zerosum","generalsum"]:
            game_data = []
            # for each gametype, create an empty DataFrame using the premade columns
            dataframe = pd.DataFrame(columns=columns)
            for horizon in range(self.planning_horizon):
                new_row_data = []
                for SOTA in ["State of the Art","Stackelberg"]:
                    # get the data of the current benchmark
                    current_data = self.database[(self.database["SOTA"]==SOTA)&(self.database["horizon"]==horizon+1)&(self.database["gametype"]==gametype)]
                    time = current_data["time"].values[0]
                    value = current_data["leader values"].values[0]
                    iteration = current_data["iterations"].values[-1]
                    num_beliefs = current_data["number_of_beliefs"].values[-1]
                    # aggregate data together
                    new_row_data = new_row_data + [time,value,iteration,num_beliefs]
                game_data.append(new_row_data)
            # populate dataframe with data 
            dataframe = dataframe.merge(pd.DataFrame(game_data, columns=columns), how='outer')
            # set indexes of dataframe
            dataframe.index = [f"{PROBLEM.NAME}({horizon})" for horizon in range(1,self.planning_horizon+1)]
            #export dataframe to csv file
            if densities == FALSE : dataframe.to_csv(f"processed_results/{gametype}_{PROBLEM.NAME}.csv",index=True)
            else : dataframe.to_csv(f"processed_results/{gametype}_{PROBLEM.NAME}_{horizon}_densities_experiment.csv",index=True)
            tables[gametype]=dataframe
            self.summary_table = dataframe
        return dataframe


    def plots(self,densities = False):
        """makes plots for each gametype that shows the performance of different algorithms ("SOTA"/"Stackelberg") on each gametype"""
        fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        colors = ['blue', 'red']
        line_widths = [2.0, 1.5]
        for idx,gametype in enumerate(["cooperative","generalsum","zerosum"]):
            data = self.database[(self.database["gametype"]==gametype) & (self.database["horizon"]==self.planning_horizon)]
            x = [i+1 for i in range(0,self.iterations)]
            for color_idx,sota in enumerate(["Stackelberg","State of the Art"]):
                y = [value for value in  data["leader values"][data["SOTA"]==sota].to_numpy()]
                axs[idx].plot(x,y,label = sota,color=colors[color_idx],linewidth=line_widths[color_idx])
                axs[idx].set_title(f"{gametype}") 
        # axs[idx].legend()
       

        # set legend
        plt.legend(loc='lower center')
            
        # Add label for all axes
        fig.text(0.5, 0.04, 'Iterations', ha='center', va='center')
        fig.text(0.05, 0.5, 'leader value', ha='center', va='center', rotation='vertical')

        # set plot title 
        fig.suptitle(f"Results for {PROBLEM.NAME} with horizon = {self.planning_horizon}")
        
        # Adjust layout
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Adjust the rect parameter as needed

        #set integer labels on the x axis
        plt.xticks(range(1, self.iterations + 1))


        # if statements for saving the resulting plot depending on the experiment type 
        if densities == False : plt.savefig(f"plots/{PROBLEM.NAME}_({self.planning_horizon}).png")
        else : plt.savefig(f"plots/{PROBLEM.NAME}_({self.planning_horizon})_densities.png")
        
        plt.show(block=False)

     
    def horizon_value_plot(self,timestep=False,densities = False):
        bar_width = 0.35
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        data = pd.DataFrame(self.database)

        if timestep == False: timestep =self.planning_horizon
        for id,gametype in enumerate(["cooperative","generalsum","zerosum"]):
            
            horizons = np.arange(timestep,step=1)
            sota_values = []
            non_sota_values = []
            x_labels = []
            colors = ['darkblue', 'maroon']

            for timestep in range(1,timestep+1):
                # data selection
                data = self.database[(self.database["gametype"]==gametype) & (self.database["horizon"]==timestep)]
                
                # get leader values from different solve methods
                sota_values.append(np.average([value for value in  data["leader values"][data["SOTA"]=="State of the Art"].to_numpy()]))
                non_sota_values.append(np.average([value for value in  data["leader values"][data["SOTA"]=="Stackelberg"].to_numpy()]))
                x_labels.append(timestep)

            # plotting
            axs[id].bar(horizons, sota_values, bar_width, label='Stackelberg',color=colors[0])
            axs[id].bar(horizons + bar_width, non_sota_values, bar_width, label='State of the art',color=colors[1])

            # labels
            axs[id].set_xlabel("Horizon")
            axs[id].set_title(f"{gametype}")
            axs[id].set_xticks(horizons+ bar_width / 2,horizons)
            axs[id].set_xticklabels(x_labels)

        fig.suptitle(f"Results for {PROBLEM.NAME}")
        fig.text(0.05, 0.5, 'leader value', ha='center', va='center', rotation='vertical')

    

        plt.legend()
        # Adjust layout
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Adjust the rect parameter as needed

        if densities == False : plt.savefig(f"horizon_plot/{PROBLEM.NAME}_({timestep}).png")
        else : plt.savefig(f"horizon_plot/{PROBLEM.NAME}_({timestep}_densities.png")
        plt.show(block=False)
        plt.pause(5)
        plt.close('all')
        return

    def density_plot(self,horizon=None):

        # Assuming belief_sizes is a list of strings or any other type you want to use as labels

        fig, axs = plt.subplots(3, 1, figsize=(9, 7))
        colors = ['darkblue', 'maroon']
        bar_width = 0.25
        data = pd.DataFrame(self.database)

        if horizon is None : horizon = self.planning_horizon

        for idx, gametype in enumerate(["cooperative", "zerosum", "generalsum"]):
            # select data 
            current_data = data[(data["gametype"] == gametype) & (data["horizon"] == horizon)]
            
            # get belief sizes used during solving of each gametype
            belief_sizes = current_data["number_of_beliefs"][current_data["SOTA"] == "State of the Art"].to_numpy()
            
            # get leader values from different solve methods
            sota_leader_values = current_data["leader values"][current_data["SOTA"] == "State of the Art"].to_numpy()
            non_sota_leader_values = current_data["leader values"][current_data["SOTA"] == "Stackelberg"].to_numpy()
            axs[idx].set_title(f"{gametype}")

            print(f"belief size: {belief_sizes},\n sota values: {sota_leader_values},\n non-sota values: {non_sota_leader_values}")

            x = np.arange(len(belief_sizes))  # Generating x-values for bars

            # Plotting
            axs[idx].bar(x - bar_width / 2, sota_leader_values, bar_width, label='Stackelberg', color=colors[0])
            axs[idx].bar(x + bar_width / 2, non_sota_leader_values, bar_width, label='State of the art', color=colors[1])

            # Setting x-axis ticks and labels
            axs[idx].set_xticks(x)
            axs[idx].set_xticklabels(belief_sizes)
            axs[idx].set_xlabel("Belief space size")

        plt.legend()

        fig.suptitle(f"Belief size plot for {PROBLEM.NAME}, Horizon = {horizon})")

        fig.text(0.05, 0.5, 'leader value', ha='center', va='center', rotation='vertical')
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # Adjust the rect parameter as needed


        # if statements for saving the resulting plot depending on the experiment type 
        plt.savefig(f"density_plot/{PROBLEM.NAME}_({horizon})_{self.iterations}.png")
        
