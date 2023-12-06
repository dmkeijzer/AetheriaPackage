import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))

import AetheriaPackage.structures as struct

# inputs
h0 = 2
b0 = 2.2
V = 1.13
number_tanks = 2

# resolution of 2D graphs
resolution = 40

# range of length of tanks
l_tank = np.linspace(1, 6, resolution)


# Plot with Beta as variable (using the 'viridis_r' colormap in reverse order)
# struct.plot_variable(h0, b0, V, l_tank, number_tanks,'Beta', np.linspace(0.4, 0.8, 60), 'ARe', 2.8)
struct.plot_variable(h0, b0, V, l_tank, number_tanks,'ARe', np.linspace(2.4, 3.1, 60), 'Beta', 0.5)
