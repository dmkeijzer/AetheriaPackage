""""Plot the history of a MADO execution"""

import numpy  as np
import os
import pandas as pd
import matplotlib.pyplot  as plt


data_aircraft = pd.read_csv(r"output\Final_design_Nov_25_15.48\aetheria_Aircraft_hist.csv")
data_wing = pd.read_csv(r"output\Final_design_Nov_25_15.48\aetheria_Wing_hist.csv")
data_fuselage = pd.read_csv(r"output\Final_design_Nov_25_15.48\aetheria_Fuselage_hist.csv")
data_aero = pd.read_csv(r"output\Final_design_Nov_25_15.48\aetheria_Aero_hist.csv")


fig, axs = plt.subplots(2,1)
fig.set_size_inches(8,6)

axs[0].plot(np.array(data_aircraft["mission_energy"])/3.6e6, "k>-", label="Mission Energy")
axs[0].set_xlabel(r"$n_{th}$ iteration")
axs[0].set_ylabel(r"E [kWh]")
axs[0].grid()

ax2 = axs[0].twinx()
ax2.plot(data_aircraft["MTOM"],  "ro-", label="MTOM", alpha=0.8)
ax2.set_ylabel("Mass [kg]")
ax2.set_xlabel(r"$n_{th}$ iteration")
axs[0].legend()

lines_1, labels_1 = axs[0].get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
axs[0].legend(lines_1 + lines_2, labels_1 + labels_2)

axs[1].plot(data_wing["aspect_ratio"], "r>-", label="Aspect ratio", alpha=0.8)
axs[1].set_ylabel(r"$A$ [-]")
ax3 = axs[1].twinx()
ax3.plot(data_fuselage["length_fuselage"], "ko-", label="Fuselage Length")
ax3.set_ylabel(r"$l_{fuse}$ [m]")
axs[1].set_xlabel(r"$n_{th}$ iteration")
axs[1].grid()

lines_1, labels_1 = axs[1].get_legend_handles_labels()
lines_2, labels_2 = ax3.get_legend_handles_labels()
axs[1].legend(lines_1 + lines_2, labels_1 + labels_2)


fig.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Downloads", "converg_hist.pdf"), bbox_inches= "tight")
