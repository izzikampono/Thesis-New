import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decpomdp import DecPOMDP
from problem import PROBLEM
import gc 
import sys
gc.enable()

# input : file_name , game type  , planning horizon, num iterations,sota(1 or 0)
# sample code  : 
# python experiment_script.py problem=dectiger horizon=10 iter=3
# or
# python experiment_script.py problem=dectiger horizon=3 iter=3 density=0.1

if len(sys.argv) < 2:
    print("err0r : not enough arguments given")
    sys.exit(1)
if len(sys.argv)> 3 :
    file_name = str(sys.argv[1]).split("=")[1]
    planning_horizon = int(sys.argv[2].split("=")[1])
    num_iterations = int(sys.argv[3].split("=")[1])
elif len(sys.argv)> 4 :
    file_name = str(sys.argv[1]).split("=")[1]
    planning_horizon = int(sys.argv[2].split("=")[1])
    num_iterations = int(sys.argv[3].split("=")[1])
    density = float(sys.argv[3].split("=")[1])
else : 
    print("not enough arguments")
    sys.exit()

#import problem and initialize
PROBLEM.initialize(DecPOMDP(file_name,horizon=planning_horizon))
from experiment import Experiment
experiment = Experiment(planning_horizon,num_iterations)



if len(sys.argv)> 3 :
    #configure experiment and run
    experiment.run_experiments(0.05)
# if statement for when density is specificed as a command line argument
else :
    experiment.run_experiments(density)
experiment.generate_comparison_tables()
experiment.generate_summary_table()
experiment.horizon_value_plot()
experiment.density_plot()
experiment.plots()


