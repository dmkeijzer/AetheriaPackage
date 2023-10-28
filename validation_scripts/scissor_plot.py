import numpy as np
import sys
import pathlib as pl

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

from scripts.power.finalPowersizing import power_system_convergences
from modules.stab_ctrl.loading_diagram import loading_diagram
import input.data_structures  as data
import input.GeneralConstants as const
from modules.stab_ctrl.aetheria_stability_derivatives_edited import downwash, downwash_k
from modules.stab_ctrl.wing_loc_horzstab_sizing import *
from modules.stab_ctrl.wing_loc_horzstab_sizing import wing_location_horizontalstab_size


path = r"C:\Users\damie\OneDrive\Desktop\Damien\DSE\AetheriaPackage\output\run_optimizaton_Oct_24_22.59\design_state_Oct_24_22.59.json"

fuselage = data.Fuselage.load(path)
aircraft = data.AircraftParameters.load(path)
WingClass = data.Wing.load(path)
Aeroclass = data.Aero.load(path)
VtailClass = data.VeeTail.load(path)
PowerClass = data.Power.load(path)
EngineClass = data.Engine.load(path)

CLh_approach = CLh_approach_estimate(VtailClass.aspect_ratio)
CLAh_approach = Aeroclass.cL_max * 0.9 # Assumes fuselage contribution negligible

l_h = fuselage.length_fuselage * (1-aircraft.wing_loc)
l_fn = aircraft.wing_loc * fuselage.length_fuselage - const.x_ac_stab_wing_bar * WingClass.chord_mac - WingClass.x_lemac
depsda = downwash(downwash_k(l_fn, WingClass.span), Aeroclass.cL_alpha, WingClass.aspect_ratio) # TODO compute downwash from functions
cg_dict, cg_dict_margin = loading_diagram(aircraft.wing_loc * fuselage.length_fuselage, fuselage.length_fuselage, fuselage, WingClass, VtailClass,aircraft, PowerClass, EngineClass )
cg_front_bar = (cg_dict_margin["frontcg"] - aircraft.wing_loc * fuselage.length_fuselage + const.x_ac_stab_wing_bar * WingClass.chord_mac)/ WingClass.chord_mac
cg_rear_bar = (cg_dict_margin["rearcg"] - aircraft.wing_loc * fuselage.length_fuselage + const.x_ac_stab_wing_bar * WingClass.chord_mac)/ WingClass.chord_mac
CLaAh = CLaAhcalc(Aeroclass.cL_alpha, fuselage.width_fuselage_outer, WingClass.span, WingClass.surface, WingClass.chord_root)

# Computing aerodynamic centre
x_ac_stab_fus1_bar = x_ac_fus_1calc(fuselage.width_fuselage_outer, fuselage.height_fuselage_outer, l_fn, CLaAh, WingClass.surface, WingClass.chord_mac)
x_ac_stab_fus2_bar = x_ac_fus_2calc(fuselage.width_fuselage_outer, WingClass.surface, WingClass.span, WingClass.quarterchord_sweep, WingClass.taper, WingClass.chord_mac)
x_ac_stab_bar = const.x_ac_stab_wing_bar + x_ac_stab_fus1_bar + x_ac_stab_fus2_bar + const.x_ac_stab_nacelles_bar

# Computing moment about the aerodynamic centre
Cm_ac_fuselage = cmac_fuselage_contr(fuselage.width_fuselage_outer, fuselage.length_fuselage, fuselage.height_fuselage_outer, Aeroclass.cL_alpha0_approach, WingClass.surface, WingClass.chord_mac, CLaAh)  # CLaAh for ctrl is different than for stab if cruise in compressible flow
Cm_ac = Aeroclass.cm_ac + const.Cm_ac_flaps + Cm_ac_fuselage + const.Cm_ac_nacelles

# computing misc variables required
beta = betacalc(const.mach_cruise)
CLah = CLahcalc(VtailClass.aspect_ratio, beta, const.eta_a_f, const.sweep_half_chord_tail)

# Creating actually scissor plot
cg_bar  = np.linspace(-1,2,1000)
m_ctrl, q_ctrl = ctrl_formula_coefs(CLh_approach, CLAh_approach, l_h, WingClass.chord_mac, const.Vh_V_2, Cm_ac, x_ac_stab_bar) # x_ac_bar for ctrl is different than for stab if cruise in compressible flow
m_stab, q_stab = stab_formula_coefs(CLah, CLaAh, depsda, l_h, WingClass.chord_mac, const.Vh_V_2, x_ac_stab_bar, const.stab_margin)
ShS_stab = m_stab * cg_bar - q_stab
ShS_ctrl = m_ctrl * cg_bar + q_ctrl

# retrieving minimum tail sizing
idx_ctrl = cg_bar == min(cg_bar, key=lambda x:abs(x - cg_front_bar))
idx_stab = cg_bar == min(cg_bar, key=lambda x:abs(x - cg_rear_bar))
ShSmin = max(ShS_ctrl[idx_ctrl], ShS_stab[idx_stab])[0]

left_limit = cg_bar[idx_ctrl]
right_limit = cg_bar[idx_stab]


plt.plot(cg_bar, ShS_stab, label="Stability")
plt.plot(cg_bar, ShS_ctrl, label="Control")
plt.hlines( ShSmin, left_limit, right_limit)
plt.legend()
plt.grid()
plt.show()

wing_location_horizontalstab_size(WingClass, fuselage, Aeroclass, VtailClass, aircraft, PowerClass, EngineClass, None, VtailClass.aspect_ratio,stepsize=0.1, plot= True)
