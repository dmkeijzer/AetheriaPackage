# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl
import json

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import matplotlib.pyplot as plt

from modules.midterm_structures import *

b = 40
h = 30
t = 3

i_xx = i_xx_solid(40,30)