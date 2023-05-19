import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

label = str(48536)

csv_files = ["output\midterm_convergence\J1_" + label + "_hist.csv",
"output\midterm_convergence\L1_" + label + "_hist.csv",
"output\midterm_convergence\W1_" + label + "_hist.csv"
]



for i in range(4):

    fig, axs = plt.subplots(1,1)

    for idx, file in enumerate(csv_files):
        data = pd.read_csv(file)

        if i == 0: 
            axs.plot(data["mtom"],"^-", label= os.path.split(file)[-1][:2])
            axs.set_ylabel("MTOM [Kg]")
            axs.set_xlabel("Iterations")

        if i == 1: 
            axs.plot(data["mission_energy"].to_numpy()/3.6e6, "^-",   label= os.path.split(file)[-1][:2])
            axs.set_ylabel("Energy [KwH]")
            axs.set_xlabel("Iterations")

        if i == 2: 
            axs.plot(data["S"].to_numpy(), "^-",  label= os.path.split(file)[-1][:2])
            axs.set_ylabel("Surface Area [m^2]")
            axs.set_xlabel("Iterations")

        if i == 3: 
            axs.plot(data["ld_cr"].to_numpy(), "^-",  label= os.path.split(file)[-1][:2])
            axs.set_ylabel(r"$\frac{L}{D}$ [-]")
            axs.set_xlabel("Iterations")

    axs.legend()
    axs.grid()

    fig.tight_layout()
    plt.show()
