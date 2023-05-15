import sys
import matplotlib.pyplot as plt
import numpy as np
from performance_analysis import mission_analysis, initial_sizing
import json

plt.rcParams.update({'font.size': 16})

sys.path.append("../../data/")

# Initial data
concept         = 3
cruising_alt    = 400#np.arange(300, 3000, 100)       # [m] Estimated cruising altitude
energy          = 381114278    # [J] Energy capacity of the aircraft
data_path       = "../data/inputs_config_" + str(concept) + ".json"
E_tots = []

# for cruising_alt in alt:
# ================== Run the initial sizing ====================
perf = initial_sizing(cruising_alt, data_path, concept)
WS, WP = perf.sizing()
perf.testing()

# ======================== Climb chart =========================
climb_analysis  = mission_analysis(data_path, cruising_alt, 360, energy, concept)
climb_analysis.climb_perf_chart()
# alt = np.arange(300, 3000, 100)
# for cruising_alt in alt:
# ===== Energy needed and distribution for normal mission ======
Energy_analysis  = mission_analysis(data_path, cruising_alt, 360, energy, concept, save_data = True)
E_tot = Energy_analysis.total_energy(300e3, pie = True)
print("Total energy needed: ", E_tot, "J")
E_tots.append(E_tot/1e6)
# plt.show()
# #plt.close('all')
# print(list(alt))
# print(E_tots)
# print('ok')
# plt.title('altitude sensitivity')
# plt.plot(alt, np.array(E_tots), '.')
# plt.show()
# print('ok')

# ================= Payload range diagram ======================
# Maximum payload weight
datafile = open(data_path, "r")

# Read data from json file
data = json.load(datafile)
datafile.close()

# Read weight data
weights = data["Structures"]
MTOM    = weights["MTOW"]/9.81
EOM     = weights["EOW"]/9.81

# Maximum payload mass
m_PL_max    = MTOM - EOM

# Range of payloads
m_PL        = np.linspace(0, m_PL_max, 100)
analysis    = mission_analysis(data_path, cruising_alt, m_PL, energy, concept)

# Different weight breakdowns
tot_weight      = analysis.W
empty_weight    = analysis.EOW
max_weight      = analysis.MTOW
payload_weight  = m_PL*9.81

ranges = analysis.range()/1e3

plt.plot(ranges, tot_weight)
plt.plot(ranges, np.ones(np.size(tot_weight))*empty_weight, color = 'black')
plt.plot(ranges, np.ones(np.size(tot_weight))*max_weight, color = 'black')
plt.fill_between(ranges, np.ones(np.size(tot_weight))*empty_weight, np.zeros(np.size(tot_weight)), color = 'red',
                 alpha = 0.5)
plt.fill_between(ranges, np.ones(np.size(tot_weight))*max_weight, np.ones(np.size(tot_weight))*1e9, color = 'red',
                 alpha = 0.5)

plt.ylim(0.98*empty_weight, 1.02*max_weight)
plt.xlim(min(ranges), max(ranges))
plt.ylabel('Aircraft weight [N]')
plt.xlabel('Cruise range [km]')
plt.tight_layout()
plt.grid()
path = 'C:/Users/Egon Beyne/Desktop/DSE/Plots/payload_range_' + str(concept) + '.pdf'
plt.savefig(path)
plt.show()
