
import numpy as np
import os
import json
import sys
import pathlib as pl
import time

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.convergence.integration import run_integration
from input.data_structures.performanceparameters import PerformanceParameters

label  = ("_".join(time.asctime().split(" ")[1:-1])).replace(":",".")[:-3]

for i in range(2):
    run_integration(label)
