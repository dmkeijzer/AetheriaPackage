
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[3]))
os.chdir(str(list(pl.Path(__file__).parents)[3]))

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")


label = str("May_22_15.44")

csv_files = ["output\midterm_sensitivity_diskloading\J1_" + label + "_sensitivity.csv",
"output\midterm_sensitivity_diskloading\L1_" + label + "_sensitivity.csv",
"output\midterm_sensitivity_diskloading\W1_" + label + "_sensitivity.csv"
]

var_list = ["mtom", "mission_energy", "S", "ld_cr", "n_ult", "e"]
unit_list = ["[kg]", "[KwH]", "[m^2]", "[-]", "[-]", "[-]"]



for file in csv_files:
    data = pd.read_csv(file)

    plt.plot(data["diskloading"], data["mtom"], "^-", label= os.path.split(file)[-1][:2])
    plt.scatter([120, 1200, 320], [1927, 4347, 3164], c= ["b", "tab:orange", "g"])
    plt.text(120+20, 1927 - 200, "J1 Midterm Design")
    plt.text(1200+20, 4347- 50, "L1 Midterm Design")
    plt.text(320+20, 3164 - 50, "W1 Midterm Design")
    plt.xlabel("Disk Loading [kg/m^2]")
    plt.ylabel(" Mtom [kg]")
    
plt.legend()
plt.grid()
plt.show()
