import matplotlib.pyplot as plt
from mission_power_energy import *
import seaborn as sns

os.chdir(str(list(pl.Path(__file__).parents)[3]))

# Generate a palette of 8 colors using the "hls" palette
colors = list(sns.color_palette("hls", n_colors=10))

dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

bardata = []
# Loop through the JSON files
for dict_name in dict_names:
    
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)

    mission = [data["takeoff_energy"], data["trans2hor_energy"], data["climb_energy"],  
               data["cruise_energy"], data["descend_energy"], data["hor_loiter_energy"],  
               data["trans2ver_energy"], data["ver_loiter_energy"], data["land_energy"]]
    for i in range(len(mission)):
        mission[i] = mission[i]/3.6e6
    bardata.append(mission)
    print(bardata)

labels = ["Take-off", "Transition to horizontal", "Climbing", "Cruise", "Descend", "Loiter horizontal","Transition to vertical", "Loiter vertical", "Landing"]

# Width of each bar
bar_width = 0.2

# X locations for the bars
x = np.arange(len(labels))

# Plotting the bar graph
for i, segments in enumerate(bardata):
    plt.bar(x + (i * bar_width), segments, width=bar_width, tick_label= labels)

plt.xlabel('Mission Segments')
plt.xticks(rotation= -45, fontsize= 8)
plt.ylabel('Energy (kWh)')

name = ["J1", "L1", "W1"]
plt.legend(labels=name, loc="upper right",fontsize=10)
plt.title("Mission Energy "+ data["name"], fontsize = 12)
plt.tight_layout()

plt.show()