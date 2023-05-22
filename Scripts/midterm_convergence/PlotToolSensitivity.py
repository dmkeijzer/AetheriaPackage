
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

write_option = int(input("Type 1 to save to your downloads folder, press 0 to simply view them one by one\n"))

label = str("May_22_09.19")

csv_files = ["output\midterm_convergence\J1_" + label + "_sensitivity.csv",
"output\midterm_convergence\L1_" + label + "_sensitivity.csv",
"output\midterm_convergence\W1_" + label + "_sensitivity.csv"
]

var_list = ["mtom", "mission_energy", "S", "ld_cr", "n_ult", "e"]
unit_list = ["[kg]", "[KwH]", "[m^2]", "[-]", "[-]", "[-]"]



for file in csv_files:
    data = pd.read_csv(file)

    plt.plot(data["diskloading"], data["mtom"], label= os.path.split(file)[-1][:2])
    plt.xlabel("Disk Loading [kg/m^2]")
    plt.ylabel(" Mtom [kg]")
    
if write_option:
    plt.savefig(os.path.join(download_dir,  label + "_" + var +  ".pdf"))
else: 
    plt.show()
