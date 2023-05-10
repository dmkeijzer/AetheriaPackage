import Propulsion as prop
from constants import *
import ActuatorDisk as AD
import numpy as np
import Propulsion_trade_off as PTO
import Aero_tools as AT
import battery as bat

g0 = 9.80665
MTOM = MTOW/g0
ISA = AT.ISA(h_cruise)
rho = ISA.density()

sensitivity_factors = [0.5, 1, 2]
# sensitivity_factors = [1, 1, 1]

for f in sensitivity_factors:
    disk = AD.ActDisk(TW_ratio, MTOW/g0, V_e_LTO, V_cruise, MTOW/LD_ratio, f*D_loading)

    print("###################################################")
    print("############### FACTOR = ", f, "###################")
    print("###################################################")

    if Prop_config == 1 or Prop_config == 2:
        print("--- ACTUATOR DISK THEORY ---")
        disk_A_per_prop = disk.A/N_hover
        print("Disk area per propeller:", disk_A_per_prop, "[m^2]")
        r_out = np.sqrt(disk_A_per_prop / (np.pi*(1-D_inner_ratio**2)))

        print(" ")
        print("--- Power ---")
        P_cr = prop.PropulsionCruise(MTOM, N_cruise, disk_A_per_prop, eff_P_cr, eff_D_cr, eff_F_cr, eff_M_cr, eff_PE_cr,
                                     eff_B_cr, rho, V_cruise, MTOW / LD_ratio)
        P_h = prop.PropulsionHover(MTOM, N_hover, disk_A_per_prop, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                   disk.v_e_hover(), 0, rho, Ducted)

        print("The power needed for cruise is:", P_cr.P_cr() * 1.2, "[W]")
        print("The power needed for hover is:", P_h.P_hover() * 1.2, "[W]")
        print(" ")

        print("--- Energy ---")
        time_cruise = 300 * 1000 / V_cruise
        req_energy = P_h.P_hover() * 1.2 * (4 / 60) / 1000 + P_cr.P_cr() * 1.2 * (time_cruise / 3600) / 1000
        print("Total energy for the mission:", req_energy, "[kWh]")
        print(" ")

        print("--- P for max thrust ---")
        P_max = prop.PropulsionHover(f * MTOM * TW_ratio, N_hover, disk_A_per_prop, eff_D_h, eff_F_h, eff_M_h,
                                                  eff_PE_h, eff_B_h, disk.v_e_hover(), 0, rho, Ducted)

        # P_for_max_T_engine = P_max.P_hover()
        print("The max power needed for TW ratio is:", P_max.P_hover() * 1.2, "[W]")
        print(" ")

    # Engine sizing for config 3:
    xc_wing_eng_start = 0.2
    xc_wing_eng_end = 0.8
    xb_wing_eng_start = 0.2
    taper = c_t/c_r
    b = np.sqrt(2*AR*S_front)

    if Prop_config == 3:
        print("--- ACTUATOR DISK THEORY ---")
        print("--- Engine sizes for config 3 ---")

        r_out_wing_eng = ((xc_wing_eng_end-xc_wing_eng_start)*c_r/2 - (xc_wing_eng_end-xc_wing_eng_start)*(1-taper)*c_r*xb_wing_eng_start/2) / \
                         (1 + 2*(xc_wing_eng_end-xc_wing_eng_start)*(1-taper)/b**2)

        wing_prop_hub_ratio = 0.2
        area_wing_prop = np.pi * (r_out_wing_eng**2 - (wing_prop_hub_ratio*r_out_wing_eng)**2)
        print("The area of each of the wing engines is:", area_wing_prop, "[m^2]")
        area_tilt_eng = (disk.A - 2*area_wing_prop)/4
        print("The area of each of the tilting engines is:", area_tilt_eng, "[m^2]")

        r_out = np.sqrt(area_tilt_eng / (np.pi * (1 - D_inner_ratio**2)))

        # print("--- Power ---")
        area_ratio_tilt = 4*area_tilt_eng/disk.A
        mass_tilt_eng = area_ratio_tilt*MTOM
        mass_wing_eng = MTOM-mass_tilt_eng

        P_cr_tilt = prop.PropulsionCruise(mass_tilt_eng, N_cruise, area_tilt_eng, eff_P_cr, eff_D_cr, eff_F_cr, eff_M_cr,
                                          eff_PE_cr, eff_B_cr, rho, V_cruise, MTOW/LD_ratio)

        P_h_tilt = prop.PropulsionHover(mass_tilt_eng, 4, area_tilt_eng, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                        disk.v_e_hover(), 0, rho, Ducted)
        P_h_wing = prop.PropulsionHover(mass_wing_eng, 2, area_wing_prop, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                        disk.v_e_hover(), 0, rho, Ducted)
        P_hover = P_h_wing.P_hover() + P_h_tilt.P_hover()

        print("The power needed for cruise is:", P_cr_tilt.P_cr() * 1.2, "[W]")
        print("The power needed for hover is:", P_hover * 1.2, "[W]")
        print(" ")

        print("--- Energy ---")
        time_cruise = 300*1000/V_cruise
        req_energy = P_hover * 1.2 * (4/60) / 1000 + P_cr_tilt.P_cr() * 1.2 * (time_cruise/3600) / 1000
        print("Total energy for the mission:", req_energy, "[kWh]")
        print(" ")

        P_h_tilt = prop.PropulsionHover(mass_tilt_eng, 4, area_tilt_eng, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                        disk.v_e_hover(), 0, rho, Ducted)
        P_h_wing = prop.PropulsionHover(mass_wing_eng, 2, area_wing_prop, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                        disk.v_e_hover(), 0, rho, Ducted)
        P_hover = P_h_wing.P_hover() + P_h_tilt.P_hover()

        print("--- P for max thrust ---")
        P_max_tilt = prop.PropulsionHover(f*TW_ratio*mass_tilt_eng, 4, area_tilt_eng, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                          disk.v_e_hover(), 0, rho, Ducted)
        P_max_wing = prop.PropulsionHover(f*TW_ratio*mass_wing_eng, 2, area_wing_prop, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                          disk.v_e_hover(), 0, rho, Ducted)
        P_for_max_T_3 = P_max_tilt.P_hover() + P_max_wing.P_hover()
        print("The max power needed for TW ratio is:", P_for_max_T_3 * 1.2, "[W]")
        print(" ")

    battery = bat.Battery(500, 1000, req_energy*1000, 1)
    print("The required battery mass is:", battery.mass(), "[kg]")
    print("The required battery volume is:", battery.volume(), "[m^3]")
    print(" ")





