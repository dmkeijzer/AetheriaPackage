import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib as pl
import os
import json
os.chdir(str(list(pl.Path(__file__).parents)[2]))

from mission_power_energy import *
import seaborn as sns

os.chdir(str(list(pl.Path(__file__).parents)[2]))

# Generate a palette of 8 colors using the "hls" palette
colors = list(sns.color_palette("hls", n_colors=10))

dict_directory = os.path.realpath("input/data_structures")
dict_names = ["aetheria_constants converged.json"]
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")


with open(r"input\data_structures\aetheria_constants converged.json") as jsonFile:
    data = json.load(jsonFile)

bardata = []




mission = [3, 3.7 , 30.5, 123, 5.3, 37.1, 3.7, 4.6]
bardata.append(mission)
print(bardata)

labels = ["Take-off", "Transition to horizontal", "Climbing", "Cruise", "Descend", "Loiter horizontal","Transition and landing", "Loiter vertical"]

# Width of each bar
bar_width = 0.2

# X locations for the bars
x = 0.4*np.arange(len(labels))

# Plotting the bar graph
for i, segments in enumerate(bardata):
    plt.bar(x + (i * bar_width), segments, width=bar_width, tick_label= labels)

# plt.xlabel('Mission Segments')
plt.xticks(rotation= -45, fontsize= 12)
plt.ylabel('Energy (kWh)', fontsize = 12)
name = ["Aetheria"]
plt.legend(labels=name, loc="upper right",fontsize=14)
# plt.title("Mission Energy "+ data["name"], fontsize = 12)
plt.tight_layout()

plt.show()