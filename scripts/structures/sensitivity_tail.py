import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import modules.structures.fuselage_length as fl

# inputs
h0 = 1.8
b0 = 1.6
V = 0.5

# resolution of 2D graphs
resolution = 10

# range of length of tanks
l_tank = np.linspace(0.3, 6, resolution)

# Plot with ARe as variable (using the 'YlOrRd_r' colormap in reverse order)
fl.plot_variable(h0, b0, V, l_tank, 'ARe', np.linspace(1, 3, resolution), 'Beta', 0.8)

# Plot with Beta as variable (using the 'viridis_r' colormap in reverse order)
fl.plot_variable(h0, b0, V, l_tank, 'Beta', np.linspace(0.3, 0.8, resolution), 'ARe', 2)
