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

# # Define values for parameters
# # D_inner_ratio = 0.10  # ratio of inner diameter compared to outer diameter
# D_prop_pure_hover = 0.6  # ratio of pure hover prop (just config 3) compared to root chord.
#
# ActDisk = prop.ActuatorDisk(D_inner_ratio, D_prop_pure_hover)
#
# # ActDisk = ActuatorDisk(D_inner_ratio, n_prop,TWRatio,V_e_LTO)
# print("Required total area for hover", ActDisk.A_hover(), "[m**2]")
# print("Required diameter per prop is", ActDisk.D_prop_outer(), "[m] for the cruise props")
# if N_cruise != N_hover:
#     D_hover = D_prop_pure_hover * c_r
#     print("Diameter dedicated hover engines:", D_hover, "[m]")
# print("There are", N_hover, "propellers, of which", N_hover - N_cruise, "are purely for hover")
# print()
# print("Exit speed in cruise:", ActDisk.V_e_cruise(), "[m/s]")
# print("Ideal power for cruise:", ActDisk.P_ideal(), "[W]")
# print("Actual power for cruise:", ActDisk.P_actual(), "[W]")
# print("Efficiency in cruise:", ActDisk.eff(), "[-]")

disk = AD.ActDisk(TW_ratio, MTOW/g0, V_e_LTO, V_cruise, MTOW/LD_ratio, D_loading)

if Prop_config == 1 or Prop_config == 2:
    print("--- ACTUATOR DISK THEORY ---")
    # print("Max take-off mass:", MTOW/g0)
    # print("Total disk area:", disk.A_disk(), "[m^2]")
    print("Total disk area from DL:", disk.A, "[m^2]")
    # print("Equivalent radius for one propeller:", np.sqrt(disk.A_disk()/np.pi), "[m]")
    print("Equivalent radius for one propeller (from Disk Loading):", np.sqrt(disk.A/np.pi), "[m]")
    print("Propeller exit speed at hover:", disk.v_e_hover(), "[m/s]")
    print(" ")

    # print("Disk area per propeller:", disk.A_disk()/N_hover, "[m^2]")
    # r_out = np.sqrt((disk.A_disk()/N_hover) / (np.pi*(1-D_inner_ratio**2)))
    disk_A_per_prop = disk.A/N_hover
    print("Disk area per propeller:", disk_A_per_prop, "[m^2]")
    r_out = np.sqrt(disk_A_per_prop / (np.pi*(1-D_inner_ratio**2)))
    print("Outer radius of the propellers:", r_out, "[m]")
    print("Hub radius of propellers:", r_out*D_inner_ratio, "[m]")
    print(" ")

    print("Cruise speed:", V_cruise, "[m/s]")
    print("Jet speed cruise:", disk.v_e_cr(), "[m/s]")
    print("Cruise propulsive efficiency:", disk.eff_cruise(), "[-]")
    print(" ")

    print("Ideal power for cruise:", disk.P_ideal(), "[W]")
    print("Actual power for cruise:", disk.P_actual(), "[W]")
    print(" ")

    print("Max thrust per engine:", MTOW*2.5/N_hover)
    print(" ")

    print("--- Power ---")
    # area_ratio_tilt = 4 * area_tilt_eng / disk.A

    P_cr = prop.PropulsionCruise(MTOM, N_cruise, disk_A_per_prop, eff_P_cr, eff_D_cr, eff_F_cr, eff_M_cr, eff_PE_cr,
                                 eff_B_cr, rho, V_cruise, MTOW / LD_ratio)
    P_h = prop.PropulsionHover(MTOM, N_hover, disk_A_per_prop, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                               disk.v_e_hover(), 0, rho, Ducted)
    print("eff_hov:", P_h.eff_hover)
    print("eff_cr:", P_cr.eff_cruise)
    print("The power needed for cruise is:", P_cr.P_cr() * 1.2, "[W]")
    print("The power needed for hover is:", P_h.P_hover() * 1.2, "[W]")
    print(" ")

    print("--- P for max thrust ---")
    P_max = prop.PropulsionHover(MTOM * TW_ratio, N_hover, disk_A_per_prop, eff_D_h, eff_F_h, eff_M_h,
                                 eff_PE_h, eff_B_h, disk.v_e_hover(), 0, rho, Ducted)

    # P_for_max_T_engine = P_max.P_hover()
    print("The max power needed for TW ratio is:", P_max.P_hover() * 1.2, "[W]")
    print(" ")

    print("--- Energy ---")
    print("Energy for hover (assuming 4 minutes in total for a flight):", P_h.P_hover() * 1.2 * (4 / 60) / 1000,
          "[kWh]")
    time_cruise = 300 * 1000 / V_cruise
    print("Energy for cruise (assuming 300 km of flight at V_Cruise):", P_cr.P_cr() * 1.2 * (time_cruise / 3600) / 1000,
          "[kWh]")
    req_energy = P_h.P_hover() * 1.2 * (4 / 60) / 1000 + P_cr.P_cr() * 1.2 * (time_cruise / 3600) / 1000
    print("Total energy for the mission:", req_energy, "[kWh]")
    print(" ")

    print("--- Thrust ---")
    print("Max thrust per engine:", TW_ratio*MTOW/N_hover, "[N]")

# Engine sizing for config 3:
xc_wing_eng_start = 0.2
xc_wing_eng_end = 0.8
xb_wing_eng_start = 0.2
taper = c_t/c_r
b = np.sqrt(2*AR*S_front)

if Prop_config == 3:
    print("--- ACTUATOR DISK THEORY ---")
    print("--- Engine sizes for config 3 ---")
    print("Total disk area from DL:", disk.A, "[m^2]")
    # print("Equivalent radius for one propeller:", np.sqrt(disk.A_disk()/np.pi), "[m]")
    print("Equivalent radius for one propeller (from Disk Loading):", np.sqrt(disk.A / np.pi), "[m]")
    print("Propeller exit speed at hover:", disk.v_e_hover(), "[m/s]")

    r_out_wing_eng = ((xc_wing_eng_end-xc_wing_eng_start)*c_r/2 - (xc_wing_eng_end-xc_wing_eng_start)*(1-taper)*c_r*xb_wing_eng_start/2) / \
                     (1 + 2*(xc_wing_eng_end-xc_wing_eng_start)*(1-taper)/b**2)
    print("The outer radius of the wing propeller is:", r_out_wing_eng, "[m]")

    wing_prop_hub_ratio = 0.2
    area_wing_prop = np.pi * (r_out_wing_eng**2 - (wing_prop_hub_ratio*r_out_wing_eng)**2)
    print("The area of each of the wing engines is:", area_wing_prop, "[m^2]")

    area_tilt_eng = (disk.A - 2*area_wing_prop)/4
    print("The area of each of the tilting engines is:", area_tilt_eng, "[m^2]")

    r_out = np.sqrt(area_tilt_eng / (np.pi * (1 - D_inner_ratio**2)))
    print("The outer radius of each of the tilting engines is:", r_out, "[m]")
    print("Hub radius of propellers:", r_out * D_inner_ratio, "[m]")

    print("Cruise speed:", V_cruise, "[m/s]")
    # print("v0 for hover:", disk.v_0_hover(), "[m/s]")
    print("Jet speed cruise:", disk.v_e_cr(), "[m/s]")
    print("Cruise propulsive efficiency:", disk.eff_cruise(), "[-]")
    print("Ideal power for cruise:", disk.P_ideal(), "[W]")
    print("Actual power for cruise:", disk.P_actual(), "[W]")
    print(" ")

    print("--- Power ---")
    area_ratio_tilt = 4*area_tilt_eng/disk.A
    mass_tilt_eng = area_ratio_tilt*MTOM
    mass_wing_eng = MTOM-mass_tilt_eng
    print("Max thrust of each wing engine:", MTOW*(1-area_ratio_tilt)*TW_ratio/2, "[N]")
    print("Max thrust of each tilt engine:", MTOW*area_ratio_tilt*TW_ratio/4, "[N]")

    P_cr_tilt = prop.PropulsionCruise(mass_tilt_eng, N_cruise, area_tilt_eng, eff_P_cr, eff_D_cr, eff_F_cr, eff_M_cr,
                                      eff_PE_cr, eff_B_cr, rho, V_cruise, MTOW/LD_ratio)
    # P_cr_wing = prop.PropulsionCruise(mass_wing_eng, 0, area_wing_prop, eff_P_cr, eff_D_cr, eff_F_cr, eff_M_cr,
    #                                   eff_PE_cr, eff_B_cr, rho, V_cruise, MTOW/LD_ratio)

    P_h_tilt = prop.PropulsionHover(mass_tilt_eng, 4, area_tilt_eng, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                    disk.v_e_hover(), 0, rho, Ducted)

    P_h_wing = prop.PropulsionHover(mass_wing_eng, 2, area_wing_prop, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                    disk.v_e_hover(), 0, rho, Ducted)
    P_hover = P_h_wing.P_hover() + P_h_tilt.P_hover()

    print("The power needed for cruise is:", P_cr_tilt.P_cr() * 1.2, "[W]")
    print("The power needed for hover is:", P_hover * 1.2, "[W]")
    print(" ")

    print("--- P for max thrust ---")
    P_max_tilt = prop.PropulsionHover(TW_ratio * mass_tilt_eng, 4, area_tilt_eng, eff_D_h, eff_F_h, eff_M_h,
                                      eff_PE_h, eff_B_h, disk.v_e_hover(), 0, rho, Ducted)
    P_max_wing = prop.PropulsionHover(TW_ratio * mass_wing_eng, 2, area_wing_prop, eff_D_h, eff_F_h, eff_M_h,
                                      eff_PE_h, eff_B_h, disk.v_e_hover(), 0, rho, Ducted)
    P_for_max_T_3 = P_max_tilt.P_hover() + P_max_wing.P_hover()
    print("The max power needed for TW ratio is:", P_for_max_T_3 * 1.2, "[W]")
    print(" ")

    print("--- Energy ---")
    print("Energy for hover (assuming 4 minutes in total for a flight):", P_hover * 1.2 * (4/60) / 1000, "[kWh]")
    time_cruise = 300*1000/V_cruise
    print("Energy for cruise (assuming 300 km of flight at V_Cruise):", P_cr_tilt.P_cr() * 1.2 * (time_cruise/3600) / 1000, "[kWh]")
    req_energy = P_hover * 1.2 * (4/60) / 1000 + P_cr_tilt.P_cr() * 1.2 * (time_cruise/3600) / 1000
    print("Total energy for the mission:", req_energy, "[kWh]")
    print(" ")

battery = bat.Battery(500, 1000, req_energy*1000, 1)
print("The required battery mass is:", battery.mass(), "[kg]")
print("The required battery volume is:", battery.volume(), "[m^3]")
print(" ")

print("--- Effects of distributed propulsion ---")
# Formulas used for trade-off
if Prop_config == 1:
    LE_prop = PTO.LE_prop(disk.v_e_cr(), V_cruise, MTOW, S_front)
    print("The initial area was:", LE_prop.S, "[m^2]")
    print("With leading edge distributed propulsion, this area can be reduced to", LE_prop.S1(), "[m^2]")
    print("This corresponds to a ratio of", LE_prop.S_ratio())
    print(" ")
    print("The cruise speed is:", LE_prop.v0, "[m/s]")
    print("The jet speed is:", LE_prop.ve, "[m/s]")
    print("The speed ratio is:", LE_prop.ve/LE_prop.v0, "[m/s]")

# This needs to be checked and added to the json files
xc_prop = 0.7
xb_prop_start = 0.17
# End is start + number of engines in half the wing times diameter times factor for clearance,
# divided by b/2 to get % of half span
xb_prop_end = xb_prop_start + (N_hover/4 * 2*r_out * 1.08)/(b/2)
print("The propulsion goes from", xb_prop_start, "to", xb_prop_end, "of the half wing, which has a span of", b/2, "[m]")

if Prop_config == 2:
    BLI_lam = PTO.BL(S_front, b, V_cruise, xc_prop, c_r, c_t/c_r, ISA.viscosity_dyn(), rho, r_out*2)
    BLI_tur = PTO.BL(S_front, b, V_cruise, xc_prop, c_r, c_t/c_r, ISA.viscosity_dyn(), rho, r_out*2,
                     laminar=False)

    BL_height_lam_inb = BLI_lam.BL_height(xb_prop_start * b/2)
    # print(BLI_lam.c(xb_prop_start * b/2))
    BL_height_lam_outb = BLI_lam.BL_height(xb_prop_end * b/2)
    # print(BLI_lam.c(xb_prop_end * b/2))
    BL_height_tur_inb = BLI_tur.BL_height(xb_prop_start * b/2)
    # print(BLI_tur.c(xb_prop_start * b/2))
    BL_height_tur_outb = BLI_tur.BL_height(xb_prop_end * b/2)
    # print(BLI_tur.c(xb_prop_end * b/2))

    print("The height of the (fully laminar) BL at the engine closest to the fuselage is", BL_height_lam_inb,
          "[m], which corresponds to BL/D ratio of", BL_height_lam_inb/(2*r_out), "for a fan diameter of", 2*r_out, "[m]")
    print("The height of the (fully laminar) BL at the engine furthest from the fuselage is", BL_height_lam_outb,
          "[m], which corresponds to BL/D ratio of", BL_height_lam_outb/(2*r_out), "for a fan diameter of", 2*r_out, "[m]")
    print("The height of the (fully turbulent) BL at the engine closest to the fuselage is", BL_height_tur_inb,
          "[m], which corresponds to BL/D ratio of", BL_height_tur_inb/(2*r_out), "for a fan diameter of", 2*r_out, "[m]")
    print("The height of the (fully turbulent) BL at the engine furthest from the fuselage is", BL_height_tur_outb,
          "[m], which corresponds to BL/D ratio of", BL_height_tur_outb/(2*r_out), "for a fan diameter of", 2*r_out, "[m]")
