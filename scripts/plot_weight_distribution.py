
from AetheriaPackage.data_structs import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np

aircraft= AircraftParameters.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
wing = Wing.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
engine = Engine.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
power = Power.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
fuselage = Fuselage.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
vtail = VeeTail.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
Stability = Stab.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")
aero = Aero.load(r"output\Final_design_Nov_25_15.48\design_state_Nov_25_15.48.json")

mpl.rcParams['font.size'] = 19

# Data for the pie chart
labels = ["Landing gear","Misc", 'Fuselage', 'Wing', 'Vtail', 'Battery', "Fuel cell", "Engines and propellors", "Tank mass", "Cooling mass"]
sizes = [aircraft.lg_mass, aircraft.misc_mass ,fuselage.fuselage_weight,wing.wing_weight, vtail.vtail_weight, power.battery_mass ,power.fuelcell_mass, engine.totalmass, power.h2_tank_mass, power.cooling_mass]  # Values for each section
print(f'Total mass = {np.sum(sizes)}')

# Create a pie chart
plt.figure(figsize=(8, 8))  # Set the size of the chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%' )  # Plotting the pie chart
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the pie chart
plt.savefig(os.path.join(os.path.expanduser("~"), "Downloads", "weight_pie_chart.pdf"), bbox_inches= "tight")

plt.cla()

labels = ['Cruise', 'Climb', 'Emergency hover and transition', "Loitering Cruise configuration"]
sizes = [aircraft.cruise_energy, aircraft.climb_energy, aircraft.hover_energy, aircraft.hor_loiter_energy ]  # Values for each section

# Create a pie chart
plt.figure(figsize=(8, 8))  # Set the size of the chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%')  # Plotting the pie chart
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the pie chart
plt.savefig(os.path.join(os.path.expanduser("~"), "Downloads", "energy_pie_chart.pdf"), bbox_inches= "tight")
