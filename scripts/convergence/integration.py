import os
import sys
import pathlib as pl
import time
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from modules.convergence.integration import run_integration

label  = ("_".join(time.asctime().split(" ")[1:-1])).replace(":",".")[:-3]

for i in range(1):
    run_integration(label)
