import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import modules.structures.fuselage_length as fl

# inputs
h0 = 1.81
b0 = 1.6
V = 0.36433
number_tanks = 2

# resolution of 2D graphs
resolution = 200

# range of length of tanks
l_tank = np.linspace(1, 6, resolution)

# Plot with ARe as variable (using the 'YlOrRd_r' colormap in reverse order)
fl.plot_variable(h0, b0, V, l_tank, number_tanks, 'ARe', np.linspace(1, 3, resolution), 'Beta', 0.5)

# Plot with Beta as variable (using the 'viridis_r' colormap in reverse order)
fl.plot_variable(h0, b0, V, l_tank, number_tanks,'Beta', np.linspace(0.1, 0.6, 40), 'ARe', 1)
