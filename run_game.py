import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decpomdp import DecPOMDP
from problem import PROBLEM
import gc 
import sys
gc.enable()

# input : file_name , game type  , planning horizon, num iterations,sota(1 or 0)
# sample : 
# python run_game.py problem=dectiger gametype=zerosum horizon=2 iter=1 sota=0 density=0.001
if len(sys.argv) < 2:
    print("err0r : not enough arguments given")
    sys.exit(1)
if len(sys.argv)> 5 :
    file_name = str(sys.argv[1]).split("=")[1]
    gametype = str(sys.argv[2]).split("=")[1]
    planning_horizon = int(sys.argv[3].split("=")[1])
    num_iterations = int(sys.argv[4].split("=")[1])
    sota = bool(int(sys.argv[5].split("=")[1]))
    density = (float(sys.argv[6].split("=")[1]))
else : 
    print("not enough arguments")
    sys.exit()

#import problem
PROBLEM.initialize(DecPOMDP(file_name,horizon=planning_horizon))

#configure experiment and run
from experiment import Experiment
experiment = Experiment(planning_horizon,num_iterations,algorithm="maxplane")
value , time = experiment.run_single_experiment()
# policy = experiment.game.extract_policy()



