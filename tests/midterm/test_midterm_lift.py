import numpy as np
import sys
import os
import pathlib as pl
# Path handling
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from  modules.aero.midterm_datcom_methods import datcom_cl_alpha

def test_datcom_cl_alpha():
    cl_alpha = datcom_cl_alpha(10,0.3, np.radians(20))
    assert np.isclose(cl_alpha, 4.858617)