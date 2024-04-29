from pickle import FALSE
import string
import numpy as np
import pandas as pd
from traitlets import Bool
from problem import PROBLEM
import time 
from pbvi import PBVI
import matplotlib.pyplot as plt
from utilities import *
PROBLEM = PROBLEM.get_instance()
import gc
gc.enable()

class Experiment():
    def __init__(self,horizon,iterations) -> None:
        self.planning_horizon = horizon
        self.iterations = iterations
        self.game = None
        self.policies = {"cooperative" : {},"zerosum" : {},"stackelberg" : {}}
        self.database = self.initialize_database()

#===== private methods ===============================================================================================

   
    def initialize_game(self,horizon,density,gametype,sota):
        """initializes game according to the benchmark chosen, solve method, and density"""
        self.game = PBVI(horizon=horizon,density=density,gametype=gametype,sota=sota)

    
    def initialize_database(self):
        """function to initiialize database that stores results of each solve of a game.
            the database records several measurements that come with a single solve of a game, i.e : gametype, SOTA (true/false),
            horizon,iterations,time , number of beleif states in the belief space, the value at initial belief, and density
        """
        database = {"gametype":[],
                        "SOTA" : [],
                        "horizon": [],
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
        self.database["gametype"].append(game_type)
        self.database["horizon"].append(horizon)
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

    def construct_stackelberg_comparison_matrix(self):
        """function to create the general-sum game comparison matrix that shows the performance of the SOTA trained agents against the Stackelberg trained agents """
        print("Constructing Stackelberg Comparison matrix..")
        start_time = time.time()

        # initialize stackelberg comparison matrix 
        matrix = {"Strong Leader" : {"Strong Follower" : None , "Blind Follower" : None},"Weak Leader" : {"Strong Follower" : None, "Blind Follower" : None }}
        
        # populate with values of evaluated policies
        matrix["Strong Leader"]["Strong Follower"] =  self.policies["stackelberg"][False].get_alpha_pairs(0,0).get_value(self.game.belief_space.initial_belief)
        matrix["Strong Leader"]["Blind Follower"] = self.game.evaluate2(0,0,self.policies["stackelberg"][False],self.policies["stackelberg"][True])
        matrix["Weak Leader"]["Blind Follower"] = self.policies["stackelberg"][True].get_alpha_pairs(0,0).get_value(self.game.belief_space.initial_belief)
        matrix["Weak Leader"]["Strong Follower"] = self.game.evaluate2(0,0,self.policies["stackelberg"][True],self.policies["stackelberg"][False])
        print(f"Done.. in {time.time()-start_time} seconds... exporting to csv..")
        
        # convert to dataframe and export to csv file
        matrix = pd.DataFrame(matrix)
        self.comparison_matrix = matrix
        matrix.to_csv(f"comparison_matrix/{PROBLEM.NAME}({self.planning_horizon}).csv", index=False)        


#===== public methods ===============================================================================================


    def run_single_experiment(self,horizon,gametype,sota,density=0.0001):
        "function to run a single experiemnt with one type of benchmark (problem,gametype) for a given number of iterations "
        
        # initialize list to hold values at initial belief and time for each iteration
        leader_values = []
        follower_values = []
        times = []
        self.initialize_game(horizon,density,gametype,sota)
        print(f"\t\t\t Solving {gametype} {PROBLEM.NAME} GAME Horizon {horizon} WITH SOTA = {self.game.sota}  ")

        #initialize game with gametype and sota 
      
        for iter in range(1,self.iterations+1):
            print(f"iteration : {iter}")
            self.game.reset()
            value,time = self.game.solve_game(density) #type:ignore
            leader_values.append(value[PROBLEM.LEADER])
            follower_values.append(value[PROBLEM.FOLLOWER])
            times.append(time)

            # add results to database
            self.add_to_database(gametype,horizon,sota,iter,time,self.game.belief_space.size(),value[PROBLEM.LEADER],value[PROBLEM.FOLLOWER],density)
                  

        # extract policy from the last iteration
        return leader_values,follower_values,times
    
    def run_single_experiment_set_densities(self,horizon,gametype:string,sota : Bool,densities : list):
        """function to run a single experiemnt with one type of benchmark (problem,gametype) for a given number of iterations.
            at each iteration, the algorithm chooses a corresponding value from the densities argument. 
        """
        
        # initialize list to hold values at initial belief and time for each iteration
        
        times = []
        belief_sizes = []

        #initialize game with gametype and sota 
        self.initialize_game(horizon,density=None,gametype=gametype,sota=sota)
        print(f"\t\t\t Solving {gametype} {PROBLEM.NAME} GAME Horizon {horizon} WITH SOTA = {self.game.sota}  ")
        for iter in range(1,self.iterations+1):
            print(f"iteration : {iter}")
            self.game.reset()
            value ,time = self.game.solve_game(densities[iter-1]) #type:ignore
    
            times.append(time)
            belief_sizes.append(self.game.belief_space.size())

            # add results to database
            self.add_to_database(gametype,horizon,sota,iter,time,self.game.belief_space.size(),value[0],value[1],densities[iter-1])
            self.export_database(experiment_type="densities")
        return
    
    def run_experiments_save_progress(self,density=0.00001):
        """run experiments for all benchmarks of a fixed problem (solves for all gametypes and SOTA mode).
           function version that exports the progress at each horizon 
        """
        
        for gametype in ["cooperative","zerosum","stackelberg"]:
            for sota in [False,True]:
                for horizon in range(1,self.planning_horizon+1):
                    # run game with set number of iterations 
                    leader_values,follower_values,times = self.run_single_experiment(horizon,gametype,sota,density)
                    
                    # save database at each iter to save progress
                    self.export_database(current_horizon=horizon)

                # save the alpha vector at the initial belief state (will be used to get the policy or get the value of the game )
                self.policies[gametype][sota] = self.game.value_function
        # construct stackelberg comparison matrix and export 
        self.construct_stackelberg_comparison_matrix()
        # convert database to dataframe
        self.database = pd.DataFrame(self.database)

        return self.database
    
    def run_experiments(self,density=0.00001):
        """run experiments for all benchmarks of a fixed problem (solves for all gametypes and SOTA mode).
           function version that exports the progress at each horizon 
        """
        
        for gametype in ["cooperative","zerosum","stackelberg"]:
            for sota in [False,True]:
                for horizon in range(1,self.planning_horizon+1):
                    # run game with set number of iterations 
                    leader_values,follower_values,times = self.run_single_experiment(horizon,gametype,sota,density)
                   

                # save the alpha vector at the initial belief state (will be used to get the policy or get the value of the game )
                self.policies[gametype][sota] = self.game.value_function
        # construct stackelberg comparison matrix and export 
        self.construct_stackelberg_comparison_matrix()
        # save databse as a csv file
        
        self.export_database(current_horizon=horizon)
        self.database = pd.DataFrame(self.database)
        return self.database
    
    def run_experiments_without_comparison(self,density=0.00001):
        """run experiments for all benchmarks of a fixed problem (solves for all gametypes and SOTA mode)"""
        
        for gametype in ["cooperative","zerosum","stackelberg"]:
            for sota in [False,True]:
                for horizon in range(1,self.planning_horizon+1):
                    # run game with set number of iterations 
                    leader_values,follower_values,times = self.run_single_experiment(horizon,gametype,sota,density)

                # save the alpha vector at the initial belief state (will be used to get the policy or get the value of the game )
                self.policies[gametype][sota] = self.game.value_function.get_alpha_pairs(0,0)
        # save databse as a csv file
        self.export_database()
        self.database = pd.DataFrame(self.database)
        return self.database
        
        
    
    def run_experiment_decreasing_density(self,starting_density):
        """run experiments for all benchmarks of a fixed problem with decreasing densities at each iterations (all gametypes and sota mode)"""
        
        densities = exponential_decrease(starting_density,self.iterations)
        
        for gametype in ["cooperative","zerosum","stackelberg"]:
            for sota in [False,True]:
                for horizon in range(1,self.planning_horizon+1):
                    # run game with set number of iterations 
                    self.run_single_experiment_set_densities(horizon,gametype,sota,densities)
                
                # save the alpha vector at the initial belief state (will be used to get the policy or get the value of the game )
                self.policies[gametype][sota] = self.game.value_function
        # construct stackelberg comparison matrix and export 
        self.construct_stackelberg_comparison_matrix()
        # save databse as a csv file
        self.database = pd.DataFrame(self.database)
        return self.database
    
    def generate_comparison_tables(self):
        """function that generates a concise table summarizing the results of all gametypes.
            The table highlights the difference of solving each gametype using 2 different solve methods : stackelberg or state of the art
        """
         
        gametypes = ["cooperative","zerosum","general-sum"]
        # create columns for each gametype with subcolumns : 
        columns = pd.MultiIndex.from_product([gametypes, ["State of the Art Leader Value", 'Stackelberg Leader Value']])
        dataframe = pd.DataFrame(columns=columns)

        game_data = []
        for horizon in range(self.planning_horizon):
            new_row_data = []
            for gametype in ["cooperative","zerosum","stackelberg"]:
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
        # set indexes of dataframe
        dataframe.index = [f"{PROBLEM.NAME}({horizon})" for horizon in range(1,self.planning_horizon+1)]
        #export dataframe to csv file
        dataframe.to_csv(f"comparison_table/{PROBLEM.NAME}_{self.planning_horizon}.csv",index=True)
        return dataframe        


    def generate_summary_table(self,densities = False):
        """function that generates a concise table summarizing the results of the experiments.
            uses data stored in self.database 
        """
        algorithms = ['State of the Art','Stackelberg']
        # create columns for each algorithm
        columns = pd.MultiIndex.from_product([algorithms, ['time', 'leader value', 'iteration']])
       
       #initialize tables 
        tables = dict.fromkeys(["cooperative","zerosum","stackelberg"])

        for gametype in ["cooperative","zerosum","stackelberg"]:
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
                    # aggregate data together
                    new_row_data = new_row_data + [time,value,iteration]
                game_data.append(new_row_data)
            # populate dataframe with data 
            dataframe = dataframe.merge(pd.DataFrame(game_data, columns=columns), how='outer')
            # set indexes of dataframe
            dataframe.index = [f"{PROBLEM.NAME}({horizon})" for horizon in range(1,self.planning_horizon+1)]
            #export dataframe to csv file
            if densities == FALSE : dataframe.to_csv(f"processed_results/{gametype}_{PROBLEM.NAME}.csv",index=True)
            else : dataframe.to_csv(f"processed_results/{gametype}_{PROBLEM.NAME}_densities_experiment.csv",index=True)
            tables[gametype]=dataframe
        return tables


    def plots(self,densities = FALSE):
        """makes plots for each gametype that shows the performance of different algorithms ("SOTA"/"Stackelberg") on each gametype"""
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        colors = ['blue', 'red']
        line_widths = [2.0, 1.5]
        for idx,gametype in enumerate(["cooperative","stackelberg","zerosum"]):
            data = self.database[(self.database["gametype"]==gametype) & (self.database["horizon"]==self.planning_horizon)]
            x = [i+1 for i in range(0,self.iterations)]
            for color_idx,sota in enumerate(["Stackelberg","State of the Art"]):
                y = [value for value in  data["leader values"][data["SOTA"]=="Stackelberg"].to_numpy()]
                axs[idx].plot(x,y,linestyle="--",label = sota,color=colors[color_idx],linewidth=line_widths[color_idx])
                axs[idx].set_xlabel("Iterations")
                axs[idx].set_ylabel("leader value")
                axs[idx].set_title(f"{gametype} game") 
            axs[idx].legend()
        fig.suptitle(f"Results for {PROBLEM.NAME} with horizon = {self.planning_horizon}")
        plt.tight_layout()
        plt.legend()
        if densities == False : plt.savefig(f"plots/{PROBLEM.NAME}_({self.planning_horizon}).png")
        else : plt.savefig(f"plots/{PROBLEM.NAME}_({self.planning_horizon})_densities.png")
        plt.show(block=False)
       
     
    def horizon_value_plot(self,density = False):
        bar_width = 0.35
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for id,gametype in enumerate(["cooperative","stackelberg","zerosum"]):
            # colors = ['red', 'tan']
            horizons = np.arange(self.planning_horizon,step=1)
            sota_values = []
            non_sota_values = []
            x_labels = []
            colors = ['blue', 'red']

            for horizon in range(1,self.planning_horizon+1):
                data = self.database[(self.database["gametype"]==gametype) & (self.database["horizon"]==self.planning_horizon)]
                sota_values.append(np.average([value for value in  data["leader values"][data["SOTA"]=="State of the Art"].to_numpy()]))
                non_sota_values.append(np.average([value for value in  data["leader values"][data["SOTA"]=="Stackelberg"].to_numpy()]))
                x_labels.append(horizon)
            # plotting
            axs[id].bar(horizons, sota_values, bar_width, label='Stackelberg',color=colors[0])
            axs[id].bar(horizons + bar_width, non_sota_values, bar_width, label='State of the art',color=colors[1])

            # labels
            axs[id].set_xlabel("Horizon")
            axs[id].set_ylabel('Leader value')
            axs[id].set_title(f"{gametype} game")
            axs[id].set_xticks(horizons+ bar_width / 2,horizons)
            axs[id].set_xticklabels(x_labels)

        fig.suptitle(f"Results for {PROBLEM.NAME}")
        plt.legend()
        plt.tight_layout()
        if densities == False : plt.savefig(f"horizon_plot/{PROBLEM.NAME}_({self.planning_horizon}).png")
        else : plt.savefig(f"horizon_plot/{PROBLEM.NAME}_({self.planning_horizon}_densities.png")
        plt.show(block=False)
        plt.pause(8)
        plt.close('all')
