import sys
import pathlib as pl
import os
import json
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))

# Import from modules and input folder
import input.data_structures.GeneralConstants  as const
from modules.aero.clean_class2drag import *
from input.data_structures import *
from input.data_structures.ISA_tool import ISA
from input.data_structures.vee_tail import VeeTail
AeroClass = Aero()
FuselageClass = Fuselage()
VTailClass = VeeTail()
HorTailClass = HorTail()
AeroClass.load()
FuselageClass.load()
VTailClass.load()
HorTailClass.load()


os.chdir(str(list(pl.Path(__file__).parents)[2]))
# import CL_cruise from json files

# Define the directory and filenames of the JSON files
TEST = int(input(
    "\n\nType 1 if you want to write the JSON data to your download folder instead of the repo, type 0 otherwise:\n"))  # Set to true if you want to write to your downloads folders instead of rep0
dict_directory = "input/data_structures"
dict_name = "aetheria_constants.json"
download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

atm = ISA(const.h_cruise)
t_cr = atm.temperature()
rho_cr = atm.density()
mhu = atm.viscosity_dyn()


# Load data from JSON file
with open(os.path.join(dict_directory, dict_name)) as jsonFile:
    data = json.load(jsonFile)


mac = data["mac"]

# General flight variables
re_var = Reynolds(rho_cr, const.v_cr, mac, mhu, const.k)
M_var = Mach_cruise(const.v_cr, const.gamma, const.R, t_cr)
Oswald_eff_var = Oswald_eff(data["A"])


# Writing to JSON file
data["e"] = Oswald_eff_var
data["depsda"] = 0.1

# Form factor
FF_fus_var = FF_fus(data["l_fuselage"], data["d_fuselage"])
FF_wing_var = FF_wing(const.toc, const.xcm, M_var, sweep_m(data["sweep_le"], const.xcm, data["c_root"], data["b"], data["taper"]))
FF_tail_var = FF_tail(const.toc_tail, const.xcm_tail, M_var, data['sweep_halfchord_h'])

# Wetted area
S_wet_fus_var = S_wet_fus(data["d_fuselage"], data["l_cockpit"], data["l_cabin"], data["l_tail"])
S_wet_wing_var = 2 * data["S"]  # from ADSEE slides
S_wet_tail_var = 2 * data["surface_vtail"]

# Miscellaneous drag
CD_upsweep_var = CD_upsweep(data["upsweep"], data["d_fuselage"], S_wet_fus_var)
CD_base_var = CD_base(M_var, const.A_base, S_wet_fus_var)

# Skin friction coefficienct
C_fe_fus_var = C_fe_fus(const.frac_lam_fus, re_var, M_var)
C_fe_wing_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)
C_fe_tail_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)

# Total cd
CD_fus_var = CD_fus(C_fe_fus_var, FF_fus_var, S_wet_fus_var)
CD_wing_var = CD_wing(C_fe_wing_var, FF_wing_var, S_wet_wing_var, data["S"])
CD_tail_var = CD_tail(C_fe_tail_var, FF_tail_var, S_wet_tail_var)
CD0_var = CD0(data["S"], VTailClass.surface, FuselageClass.length_fuselage*FuselageClass.width_fuselage_outer, CD_fus_var, CD_wing_var, CD_upsweep_var, CD_base_var, CD_tail_var, CD_flaps=0)
print(CD_wing_var)
# Lift times S
cL_tail_times_Sh = VTailClass.CL_cruise * HorTailClass.surface
cL_wing_times_S = data["cL_cruise"]*data["S"]

total_cL = (cL_wing_times_S + cL_tail_times_Sh) / (data['S'] + HorTailClass.surface)


# Summation and L/D calculation
CDi_var = CDi(data["cL_cruise"], data["A"], data["e"])
CD_var = CD(CD0_var, CDi_var)
lift_over_drag_var = lift_over_drag(total_cL, CD_var)

print("CD0_wing", CD_wing_var / data["S"])
print("CD cruise", CD_var)
print("CL cruise", total_cL)
print("L/D cruise", lift_over_drag_var)

# Writing to JSON file
AeroClass.ld_cruise = lift_over_drag_var
AeroClass.cd_cruise = CD_var
AeroClass.cd0_cruise = CD0_var
AeroClass.cd_upsweep = CD_upsweep_var
AeroClass.cd_base = CD_base_var


# ------------------------ DRAG DURING STALL -------------- 
# General flight variables
re_var = Reynolds(rho_sl, const.v_stall, mac, mhu, const.k)
M_var = Mach_cruise(const.v_stall, const.gamma, const.R, t_stall)
Oswald_eff_var = Oswald_eff(data["A"])


# Writing to JSON file
data["e"] = Oswald_eff_var
data["depsda"] = 0.1

# Form factor
FF_fus_var = FF_fus(data["l_fuselage"], data["d_fuselage"])
FF_wing_var = FF_wing(const.toc, const.xcm, M_var, sweep_m(data["sweep_le"], const.xcm, data["c_root"], data["b"], data["taper"]))
FF_tail_var = FF_tail(const.toc_tail, const.xcm_tail, M_var, data['sweep_halfchord_h'])


# Wetted area
S_wet_fus_var = S_wet_fus(data["d_fuselage"], data["l_cockpit"], data["l_cabin"], data["l_tail"])
S_wet_wing_var = 2 * data["S"]  # from ADSEE slides
S_wet_tail_var = 2 * data["surface_vtail"]

# Miscellaneous drag
CD_upsweep_var = CD_upsweep(data["upsweep"], data["d_fuselage"], S_wet_fus_var)
CD_base_var = CD_base(M_var, const.A_base, S_wet_fus_var)

# Skin friction coefficienct
C_fe_fus_var = C_fe_fus(const.frac_lam_fus, re_var, M_var)
C_fe_wing_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)
C_fe_tail_var = C_fe_wing(const.frac_lam_wing, re_var, M_var)

# Total cd
CD_fus_var = CD_fus(C_fe_fus_var, FF_fus_var, S_wet_fus_var)
CD_wing_var = CD_wing(C_fe_wing_var, FF_wing_var, S_wet_wing_var, data["S"])
CD_tail_var = CD_tail(C_fe_tail_var, FF_tail_var, S_wet_tail_var)
CD_flaps_var = CD_flaps(60)
CD0_var = CD0(data["S"], VTailClass.surface, FuselageClass.length_fuselage*FuselageClass.width_fuselage_outer, CD_fus_var, CD_wing_var, CD_upsweep_var, CD_base_var, CD_tail_var, CD_flaps_var)

print(CD_flaps_var)
# Summation and L/D calculation
CDi_var = CDi(data["cLmax_flaps60"], data["A"], data["e"])
CD_var = CD(CD0_var, CDi_var)
lift_over_drag_var = lift_over_drag(data["cLmax_flaps60"], CD_var)

print("CD0_wing", CD_wing_var / data["S"])
print("CD in stall", CD_var)
print("CD0 stall", CD0_var)
print("L/D stall", lift_over_drag_var)

# Writing to classes file
AeroClass.ld_stall = lift_over_drag_var
AeroClass.cd_stall = CD_var
AeroClass.cd0_stall = CD0_var
AeroClass.mach_stall = M_var






if TEST:
    with open(os.path.join(download_dir, dict_name), "w") as jsonFile:
        json.dump(data, jsonFile, indent=6)
else:
    with open(os.path.join(dict_directory, dict_name), "w") as jsonFile:
        json.dump(data, jsonFile, indent=6)