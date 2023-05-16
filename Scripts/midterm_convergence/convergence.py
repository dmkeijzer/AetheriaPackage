import subprocess
import sys

# Get the path of the current Python interpreter
python_executable = sys.executable

# List of Python files to execute
python_files = [
    "path/to/file1.py",
    "path/to/file2.py",
    "path/to/file3.py"
]

# Specify the number of times to loop
loop_count = 3

# Loop through the Python files multiple times
for i in range(1, loop_count+1):
    print(f"Loop {i}")
    
    for file in python_files:
        print(f"Running {file}")
        subprocess.run([python_executable, file])
        print(f"Finished running {file}")
