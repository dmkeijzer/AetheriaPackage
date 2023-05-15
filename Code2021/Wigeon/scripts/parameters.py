
"""
This file has the current estimates that will be needed to run the code before everything can be integrated
Do NOT use these constants in your class/tool files, only on the main file
Keep in mind that these values will change (probably a lot), only use them if you need estimates to run the code
"""

# ---------------------------------------------
# THIS IS CLASS I ESTIMATION
# UPDATED MTOM IS AT THE BOTTOM OF THE FILE IF YOU NEED IT
do_not_use_mtom = 1972          # maximum take-off mass from statistical data - Class I estimation
# ---------------------------------------------

S1, S2 = 5.25, 5.25  # surface areas of wing one and two
A = 10               # aspect ratio of a wing, not aircraft
taper = 0.45         #Taper ratio of both wings

n_ult = 3.2 * 1.5    # 3.2 is the max we found, 1.5 is the safety factor
Pmax = 15.25         # this is defined as maximum perimeter in Roskam, so i took top down view of the fuselage perimeter
lf = 7.2             # length of fuselage
h_fus = 1.705        # height of fuselage
w_fus = 1.38         # width of fuselage

MAC1 = 0.65
MAC2 = 0.65

c_r = 0.8748
c_t = 0.3499


# From project guide: 95 kg per pax (including luggage)
n_pax = 5                   # number of passengers (pilot included)
m_pax = 88                  # assume average mass of a passenger according to Google
cargo_m = (95-m_pax)*n_pax  # Use difference of pax+luggage - pax to get cargo mass

# Propulsion
n_prop = 16                          # number of engines
P_cr_estim = 110024/1.2 * 0.9        # Total cruise power in W
P_cr_per_engine = P_cr_estim/n_prop  # Total cruise power per engine in W

pos_fus = 3.6                         # fuselage centre of mass away from the nose
pos_lgear = 4                         # landing gear position away from the nose
pos_frontwing, pos_backwing = 0.5, 7  # positions of the wings away from the nose
m_prop = [30] * n_prop                # list of mass of engines (so 30 kg per engine with nacelle and propeller)
pos_prop = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
            6.7]  # 8 on front wing and 8 on back wing

# Weights and CGs from weight estimation
OEM = 3092.352770706565  # kg
oem_CG = 3.6  # m
MTOM = 3652.352770706565  # kg
mtom_CG = 3.60123208251845  # m

