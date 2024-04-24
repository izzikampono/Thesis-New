import subprocess
import sys
import signal
from problem import PROBLEM
from decpomdp import DecPOMDP

# to run 
# python server_script.py problem=dectiger horizon=10 iter=3 density=0.1 time=1


def handler(signum, frame):
    # This function will be called when the timeout is reached
    print("Timeout reached. Exiting...")
    
    sys.exit(1)

# Register the signal handler for SIGALRM (alarm signal)
signal.signal(signal.SIGALRM, handler)


if len(sys.argv) < 5:
    print("err0r : not enough arguments given")
    sys.exit(1)

if len(sys.argv) < 6:
    file_name = str(sys.argv[1]).split("=")[1]
    planning_horizon = int(sys.argv[2].split("=")[1])
    num_iterations = int(sys.argv[3].split("=")[1])
    density = float(sys.argv[4].split("=")[1])
    timeout_minutes = int(sys.argv[5].split("=")[1])
    

file_name = str(sys.argv[1]).split("=")[1]
planning_horizon = int(sys.argv[2].split("=")[1])
num_iterations = int(sys.argv[3].split("=")[1])
density = float(sys.argv[4].split("=")[1])
timeout_minutes = int(sys.argv[5].split("=")[1])
timeout_seconds = timeout_minutes * 60  # Convert to seconds
command = ["python", "experiment_script.py", f"file_name={file_name}",  f"h={planning_horizon}", f"num_iterations={num_iterations}", f"density={density}"]


try:
    # Set the alarm to trigger after the specified timeout
    signal.alarm(timeout_seconds)

    # Your main script logic here
    # For example, call subprocess to execute a command
    subprocess.run(command)


    # Cancel the alarm if the command completes before the timeout
    signal.alarm(0)

    print("Script execution completed within the timeout.")
except subprocess.CalledProcessError:
    # Handle subprocess errors if necessary
    print("Subprocess returned non-zero exit status.")
except Exception as e:
    # Handle other exceptions
    print(f"An error occurred: {e}")
finally:
    # Cancel the alarm in case of any exception
    signal.alarm(0)




