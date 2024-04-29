import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decpomdp import DecPOMDP
from problem import PROBLEM
import gc 
import sys
gc.enable()


""" Python script to run the experiments with over a number iteration each with a different density parameter.
    The density parameter dictates how the belief space expands belief states in the game tree.
"""

# input : file_name , game type  , planning horizon, num iterations,sota(1 or 0)
# sample code  : 
# python densities_experiment.py problem=dectiger horizon=2 iter=2  density=0.2
if len(sys.argv) < 2:
    print("err0r : not enough arguments given")
    sys.exit(1)

elif len(sys.argv)> 4 :
    file_name = str(sys.argv[1]).split("=")[1]
    planning_horizon = int(sys.argv[2].split("=")[1])
    num_iterations = int(sys.argv[3].split("=")[1])
    starting_density = (float(sys.argv[4].split("=")[1]))
else : 
    print("not enough arguments")
    sys.exit()

#import problem and initialize
PROBLEM.initialize(DecPOMDP(file_name,horizon=planning_horizon))


#configure experiment and run
from experiment import Experiment
experiment = Experiment(planning_horizon,num_iterations)
experiment.run_experiment_decreasing_density(starting_density)
experiment.generate_summary_table(densities=True)
experiment.generate_comparison_tables(density = True)
experiment.horizon_value_plot(density=True)
experiment.plots()