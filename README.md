# isabelle-stackelberg
Python code for solving Stackelberg Partially Observable Stochastic Games Under Onesideness Using the PBVI Algorithm


There are two ways to run the program

The test.ipynb notebook provides a comprehensive way to look at the functionality of the different classes in the library.


experiment_script.py runs an N number of games with all 3 gametypes and all solve modes, it uses arguments passed in the command line. 
sample command to run :
python experiment_script.py problem=dectiger gametype=zerosum horizon=2 iter=1 sota=0 density=.1

To run experiments with decreasing density values, 
