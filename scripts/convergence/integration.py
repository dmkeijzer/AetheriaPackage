
import numpy as np
import os
import json
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.convergence.integration import run_integration
from input.data_structures.performanceparameters import PerformanceParameters

for i in range(30):
    run_integration()