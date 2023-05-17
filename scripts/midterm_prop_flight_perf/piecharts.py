import matplotlib.pyplot as plt
from mission_power_energy import *
import seaborn as sns

os.chdir(str(list(pl.Path(__file__).parents)[2]))

# Generate a palette of 8 colors using the "hls" palette
colors = list(sns.color_palette("hls", n_colors=10))

dict_directory = "input"
dict_names = ["J1_constants.json", "L1_constants.json", "W1_constants.json"]
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

# Loop through the JSON files
for dict_name in dict_names:
    # Load data from JSON file
    with open(os.path.join(dict_directory, dict_name)) as jsonFile:
        data = json.load(jsonFile)

    mission = [data["takeoff_energy"], data["trans2hor_energy"], data["climb_energy"],  
               data["cruise_energy"], data["descend_energy"], data["hor_loiter_energy"],  
               data["trans2ver_energy"], data["ver_loiter_energy"], data["land_energy"]]
    labels = ["Take-off", "Transition to horizontal", "Climbing", "Cruise", "Descend", "Loiter horizontal","Transition to vertical", "Loiter vertical", "Landing"]
    plt.pie(mission, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.show()