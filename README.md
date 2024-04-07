# isabelle-stackelberg
Python code for solving Stackelberg Partially Observable Stochastic Games Under Onesideness Using the PBVI Algorithm


There are two ways to run the program

The test.ipynb notebook provides a comprehensive way to look at the functionality of the different classes in the library.

The run_game.py script can be used to run a single experiment of one game with a fixed gametype and horizon

experiment_script.py runs an n number of games with all 3 gametypes and all solve modes, it uses arguments passed in the command line. 
the line below shows the example code to run the script in the command line :

python experiment_script.py problem=dectiger gametype=zerosum horizon=2 iter=1 sota=0 density=0.001

note that iter should be > 1 to generate a plot 
