import numpy as np
import redundancy_battery_config as con

# Characteristics from the plane
V_motor = 500  # V
E_tot = 301.1111  # kWh
per_mot = 0.99

# Cell characteristics
V_cell = 3.7  # V
C_cell = 10 #5  # Ah
E_cell = V_cell * C_cell  # Wh

# inputs
n_mot = 12  # Number of motors in aircraft
n_bat_mot = 2  # Number of batteries per motor

red = con.redundancy_power(V_motor, E_tot, V_cell, C_cell, n_mot, n_bat_mot, per_mot)

N_cells = red.N_cells_tot()
print("Number of cells for required energy/power:", N_cells)

N_cells_mot = red.N_cells_mot()
print("Number of cells for providing power and energy to the motors", N_cells_mot)

N_cells_misc = red.N_cells_misc()
print("Number of cells for providing power and energy to other systems", N_cells_misc)

N_ser = red.N_ser()
print("Number of cells in series for required voltage:", N_ser)

N_par = red.N_par()
print("Number of cells in parallel when using", N_ser, "cells in series:", N_par)

N_par_new = red.N_par_new()
print("Number of cells for", n_mot * n_bat_mot, "batteries:", N_par_new)

N_cells_mot_new = red.N_cells_mot_new()
print("New number of cells required for the motors", N_cells_mot_new, "Which is:")

increase_mot = red.increase_mot()
abs_increase = red.increase()[0]
per_increase = red.increase()[1]

print("    -", abs_increase, "cells more than needed for energy")
print("    -", per_increase, "% increase in cells")

N_c_new = red.N_cells_new()
print("The total number of cells than becomes", N_c_new)

print("Single battery:", N_ser, "in series and", int(N_par_new/(n_mot * n_bat_mot)), "cells in parallel")

print("Single battery contains", E_cell,"Wh of energy")

