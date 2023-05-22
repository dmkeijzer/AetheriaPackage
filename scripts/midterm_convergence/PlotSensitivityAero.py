import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

write_option = int(input("Type 1 to save to your downloads folder, press 0 to simply view them one by one\n"))

label = str("May_22_13.30")

csv_files = ["output\midterm_sensitivity_diskloading\J1_" + label + "_sensitivityAero.csv",
"output\midterm_sensitivity_diskloading\L1_" + label + "_sensitivityAero.csv",
"output\midterm_sensitivity_diskloading\W1_" + label + "_sensitivityAero.csv"
]

var_list = ["mtom", "mission_energy", "S", "ld_cr", "n_ult", "e", "A"]
unit_list = ["[kg]", "[KwH]", "[m^2]", "[-]", "[-]", "[-]"]



for file in csv_files:
    data = pd.read_csv(file)
    if data["name"][0] == "J1":
        

        plt.plot(data["A"], data["mission_energy"], label= os.path.split(file)[-1][:2])
        plt.xlabel("Aspect ratio [-]")
        plt.ylabel(" Mission Energy [J]")
    else:
        

        plt.plot(data["A1"], data["mission_energy"], label= os.path.split(file)[-1][:2] )
        # plt.plot(data["A2"], data["mtom"], label= os.path.split(file)[-1][:2]  + " wing 2")
        plt.xlabel("Aspect ratio [-]")
        plt.ylabel(" Mission Energy [J]")

    
plt.legend()
plt.grid()
if write_option:
    plt.savefig(os.path.join(download_dir,  label + "_" + var +  ".pdf"))
else: 

    plt.show()