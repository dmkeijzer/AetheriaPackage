import matplotlib.pyplot as plt
from mission_power_energy import *
import seaborn as sns

# Generate a palette of 8 colors using the "hls" palette
colors = list(sns.color_palette("hls", n_colors=8))
mission = [E_to, E_trans_ver2hor + E_trans_hor2ver, E_climb, E_cr, E_desc, E_loit_cr, energy_landing_var]
labels = ["Take-off", "Transition", "Climbing", "Cruise", "Descend", "Loitering", "Landing"]
plt.pie(mission, labels=labels, colors=colors, autopct='%1.1f%%')
plt.show()