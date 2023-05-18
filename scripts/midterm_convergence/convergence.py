import subprocess
import sys
import pathlib as pl
import os
import numpy as np
import json
import pandas as pd

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

# Get the path of the current Python interpreter
python_executable = sys.executable

TEST = False # If True JSON gets writtien to your downloads folder, these values are not correct however
label = str(np.random.randint(0,1e5,1)[0])


# List of Python files to execute
python_files = [
    "scripts/Preliminary_sizing/wing_power_loading.py",
    "scripts/midterm_aerodynamics/planform_sizing.py",
    "scripts/midterm_prop_flight_perf/mission_power_energy.py",
    "scripts/midterm_structures/class2_weight_estimation.py"
]

dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]

output_dir = "output/midterm_convergence"
# Specify the number of times to loop
loop_count = 50

# Loop through the Python files multiple times
for i in range(1, loop_count+1):
    print(f"\n\n=====================\nLoop {i}\n=====================")
    
    for file in python_files:
        print(f"\nRunning {file}\n")
        process =  subprocess.Popen([python_executable, file], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        # Send input to the subprocess
        print(f"\nFinished running {file}\n")

        if TEST:
            process.stdin.write("1" + "\n")
        else:
            process.stdin.write("0" + "\n")
        process.stdin.flush()
        print(process.stdout.read())

    if not TEST:
        for dict_name in dict_names:
            # Load data from JSON file
            with open(os.path.join(dict_directory, dict_name)) as jsonFile:
                data = json.load(jsonFile)

            if os.path.exists(os.path.join(output_dir, dict_name[:2] + "_" + label + "_hist.csv")):
                pd.DataFrame(np.array(list(data.values())).reshape(1, len(data))).to_csv(os.path.join(output_dir, dict_name[:2] + "_" + label + "_hist.csv") , mode="a", header=False, index= False)
            else: 
                pd.DataFrame([data]).to_csv(os.path.join(output_dir, dict_name[:2] + "_" + label + "_hist.csv"), columns= list(data.keys()), index=False)
                    # Read the output from the subprocess

        # Print the output
