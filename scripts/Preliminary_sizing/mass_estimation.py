# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))



from modules.preliminary_sizing import *

mass_estimation(510, PLOT=True, PRINT=True)