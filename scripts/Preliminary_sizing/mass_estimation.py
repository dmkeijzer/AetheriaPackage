# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from modules.preliminary_sizing import *

Structure.MTOMclassI, Structure.OEMclassI = mass_estimation(Structure.payload, PLOT=True, PRINT=True)

