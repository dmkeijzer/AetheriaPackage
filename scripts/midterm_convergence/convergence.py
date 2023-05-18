import subprocess
import sys
import pathlib as pl
import os

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

# Get the path of the current Python interpreter
python_executable = sys.executable

TEST = False # If True JSON gets writtien to your downloads folder, these values are not correct however


# List of Python files to execute
python_files = [
    "scripts/Preliminary_sizing/wing_power_loading.py",
    "scripts/midterm_aerodynamics/planform_sizing.py",
    "scripts/midterm_prop_flight_perf/mission_power_energy.py",
    "scripts/midterm_structures/class2_weight_estimation.py"
]

# Specify the number of times to loop
loop_count = 10

# Loop through the Python files multiple times
for i in range(1, loop_count+1):
    print(f"\n\n=====================\nLoop {i}\n=====================")
    
    for file in python_files:
        print(f"\nRunning {file}\n")
        process =  subprocess.Popen([python_executable, file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        # Send input to the subprocess

        if TEST:
            process.stdin.write("1" + "\n")
        else:
            process.stdin.write("0" + "\n")
        process.stdin.flush()

        # Read the output from the subprocess
        print(process.stdout.read())

        # Print the output
        print(f"\nFinished running {file}\n")
