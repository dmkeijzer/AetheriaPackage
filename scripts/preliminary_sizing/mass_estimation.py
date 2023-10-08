# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

from modules.preliminary_sizing import *
import input.GeneralConstants as const


MTOMclassI, OEMclassI = mass_estimation(const.m_pl, PLOT=True, PRINT=True)

