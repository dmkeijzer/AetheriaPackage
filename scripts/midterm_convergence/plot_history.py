import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

write_option = int(input("Type 1 to save to your downloads folder, press 0 to simply view them one by one\n"))

label = str("May_21_13.42")

csv_files = ["output\midterm_convergence\J1_" + label + "_hist.csv",
"output\midterm_convergence\L1_" + label + "_hist.csv",
"output\midterm_convergence\W1_" + label + "_hist.csv"
]

var_list = ["mtom", "mission_energy", "S", "ld_cr", "n_ult"]
unit_list = ["[kg]", "[KwH]", "[m^2]", "[-]", "[-]"]


for var, unit in zip(var_list, unit_list):

    fig, axs = plt.subplots(1,1)

    for file in csv_files:
        data = pd.read_csv(file)

        if var == "mission_energy":
            axs.plot(data[var]/3.6e6,"^-", label= os.path.split(file)[-1][:2])
        else:
            axs.plot(data[var],"^-", label= os.path.split(file)[-1][:2])
        axs.set_ylabel(" ".join(var.split("_")) + " " +  unit)
        axs.set_xlabel("Iterations")

    axs.legend()
    axs.grid()
    
    if write_option:
        plt.savefig(os.path.join(download_dir,  label + "_" + var +  ".pdf"))
    else: 
        plt.show()
