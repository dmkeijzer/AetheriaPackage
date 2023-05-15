import numpy as np

# Physics
g = 9.80665                 # [m/s^2]
gamma = 1.4                 # [-]
R = 287                     # [J/kg/K]
rho_0 = 1.225               # [kg/m^3]

# Contingency
energy_cont     = 1
payload_cont    = 1
mass_cont       = 1.1

# Rotation mechanism constants
y_tilt_1 = 0.8865     # [m] position of the tilting wing root
y_tilt_2 = 0.8865

c_rot_1 = 0.25      # [-] position of the rotating axis in percentage chord
c_rot_2 = 0.25


# General
mission_range = 300e3       # [m] Mission range  TODO: Maybe add 50 km

n_pax = 5                               # Number of passengers
m_pax = 88                              # Max per pax
m_cargo_per_pax = 7                     # [kg] Cargo mass per pax
m_cargo_tot = m_cargo_per_pax*n_pax     # [kg] Total cargo mass
x_f_pax = 3.25      # [m] front pax position
x_r_pax = 4.75      # [m] rear pax position
x_pil = 1.75        # [m] pilot position

# Fuselage
w_fuselage = 1.38            # [m]
h_fuselage = 1.7             # [m]
l_nosecone = 2.5             # [m]
l_cylinder = 2.5             # [m]
l_tailcone = 2.7            # [m]

cargo_pos = [6.5, 0, 0.3*h_fuselage]       # [m] Cargo position

# Aerodynamics
sweepc41 = 0                # Sweep angle at quarter chord for 1st wing [rad]
sweepc42 = 0                # Sweep angle at quarter chord for the 2nd wing
k = 0.634 * 10**(-5)        # Smooth paint from adsee 2 L2  smoothness factor[-]
flamf = 0.1                 # From ADSEE 2 L2 GA aircraft [-]
IF_f = 1                    # From ADSEE 2 L2 Interference factor fuselage [-]
IF_w = 1.1                  # From ADSEE 2 L2 Interference factor wing [-]
IF_v = 1.04                 # From ADSEE 2 L2 Interference factor vertical tail [-]
flamw = 0.35                # From ADSEE 2 L2 GA aircraft
Abase = 0                   # Base area of the fuselage [m2]
tc = 0.12                   # NACA0012 for winglets and Vtail [-]
tc_wing = 0.17              # t/c for wing
xcm = 0.3                   # NACA0012 for winglets and Vtail [-]
k_wl = 2.0                  # Constant for winglets (could be changed to 2 if we need extra eff)
Vr_Vf_2 = 1                 # Speed ratio between wing 1 and 2
e_f = 0.65                  # Oswald efficiency for front wing
e_r = 0.65                  # Oswald efficiency for rear wing
h_wt_1 = 0.5                    # Height of front wingtips [m]
h_wt_2 = 0.5                    # Height of back wingtips [m]
dihedral1 = np.deg2rad(-0.5)    # Dihedral front wing. Old 5
dihedral2 = np.deg2rad(-4)      # Dihedral back wing. Old 0
i1 = 0                          # Trim angle of the front wing TODO if used deg or rad?

# Propulsion
xi_0 = 0.1         # Dimensionless radius of the hub (r_hub/R)
c_fp = 0.3         # [m] Horizontal clearance between the widest part of the fuselage and the radius of the inboard prop
c_pp = 0.3         # [m] Horizontal clearance between the propellers (closest point, tip to tip)
eff_prop = 0.85    # [-] Propeller efficiency during normal flight
eff_hover = 0.75   # [-] Propeller efficiency during hover
# TODO: revise
# eff_eng_bat = 0.7  # [-] Efficiency from batteries to engines (including engine, battery, and electronics efficiencies)
sp_mass_en = 1/5000     # [kg/W]

n_prop_1 = 6                # number of propellers on the front wing
n_prop_2 = 6                # number of propellers on the rear wing
n_prop = n_prop_1+n_prop_2  # Total number of propellers

# Power
sp_en_den = 500          # [Wh/kg] Specific energy density
vol_en_den = 900         # [Wh/l] Volumetric energy density
bat_cost = 100           # [$/kWh] Cost of batteries in US dollars per kilogram
DoD = 0.8                # [-] Depth of Discharge of the total battery
P_den = 6500             # [W/kg] Power density of battery
EOL_C = 0.85             # [-] Fraction of initial capacity that is available at end-of-life
eff_bat_eng_cr = 0.9     # Efficiency from the battery to the engines (including both) in cruise
eff_bat_eng_h = 0.75     # Efficiency from the battery to the engines (including both) in hover

# Stability
fus_back_bottom = [6.5, 0]
fus_back_top = [7.5, h_fuselage]
turn_over = np.radians(55)      # Turn-over angle
pitch_lim = np.radians(20)      # Pitch limit
lat_lim = np.radians(5)         # lateral ground clearance angle
min_ng_load = 0.1               # minimum fraction of the total weight to be carried by the nose gear
b_max       = 11
elev_fac = 1.3
crmaxf = 2.1
crmaxr = 3
A_range_f = [5, 15]
A_range_r = [5, 15]
ARv = 1.4                       # AR of vertical tail
sweep_vtail = np.deg2rad(25)    # Put in degrees and convert to rad
br_bv = 1.00                    # Span of rudder wrt span tail
cr_cv = 0.24                    # Ratio of the chords
x_ng    = x_pil                 # Nosegear position
x_tg    = cargo_pos[0]          # Tailgear position
tw_ng = 1*w_fuselage              # [m] Track width of the fuselage
sweep_vtail_c4 = 41.11158       # sweep =  41.11158066620898 deg
wing_clearance_aft = 0.3        # [m] Distance to be left behind the rear wing so it can be placed as high as possible

TW_ratio_control = 1.5    # Thrust-to-weight ratio needed for the aircraft to be controllable in hover

# Structures
n_ult = 3.4 * 1.5           # 3.2 is the max we found, 1.5 is the safety factor

