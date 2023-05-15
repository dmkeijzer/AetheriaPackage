import Noise as brr
from constants import *
import Aero_tools as at
import numpy as np
import ActuatorDisk as AD
import Propulsion as prop

# ISA + Earth thingies
g0 = 9.80665
ISA = at.ISA(h_cruise)
a = ISA.soundspeed()
rho = ISA.density()

disk = AD.ActDisk(TW_ratio, MTOW/g0, V_e_LTO, V_cruise, MTOW/LD_ratio, D_loading)

# Engine sizing for config 3:
xc_wing_eng_start = 0.2
xc_wing_eng_end = 0.8
xb_wing_eng_start = 0.2
taper = c_t/c_r
b = np.sqrt(AR*S_front)
MTOM = MTOW/g0

if Prop_config == 1:
    M_t_max = 0.7

    disk_A_per_prop = disk.A/N_hover
    r_out = np.sqrt(disk_A_per_prop / (np.pi*(1-D_inner_ratio**2)))
    # print("Disk area per propeller:", disk_A_per_prop, "[m^2]")
    # print("Outer radius of the propellers:", r_out, "[m]")
    # print(" ")

    # Calculate available rpm based on allowable Mach at the blade tips
    rpm_max = M_t_max*a*60 / (np.pi * 2*r_out)
    print("For config 1, the maximum allowable rpm are:", rpm_max, "[rpm]")
    print("")

    # Shaft power of each engine in kW
    P_br_cruise = P_cr_estim/N_cruise * 1/1000 * eff_B_cr * eff_PE_cr * eff_M_cr / 1.2
    P_br_hover = P_hover_estim/N_hover * 1/1000 * eff_B_h * eff_PE_h * eff_M_h / 1.2
    print("jdwndinwi:", 0.65 / (eff_B_h * eff_PE_h * eff_M_h))
    # The 1s are number of propellers (we calculate individually each engine, so 1)
    noise = brr.Noise(P_br_cruise, P_br_hover, 2*r_out, num_blades, 1, 1, rpm_max, rpm_max, a, M_t_h=M_t_max, M_t_cr=M_t_max)

    print("Noise level in cruise per engine:", noise.SPL_cr(), "[dB]")
    print("Noise level in cruise, total:", brr.sum_noise(np.ones((1, N_cruise))[0] * noise.SPL_cr()), "[dB]")
    print("")

    print("Noise level in hover per engine:", noise.SPL_hover(), "[dB]")
    print("Noise level in hover, total:", brr.sum_noise(np.ones((1, N_hover))[0] * noise.SPL_hover()), "[dB]")
    print("")

if Prop_config == 2:
    M_t_max = 0.55

    disk_A_per_prop = disk.A/N_hover
    r_out = np.sqrt(disk_A_per_prop / (np.pi*(1-D_inner_ratio**2)))

    # Shaft power of each engine in kW
    P_br_cruise = P_cr_estim/N_cruise * 1/1000 * eff_B_cr * eff_PE_cr * eff_M_cr / 1.2
    P_br_hover = P_hover_estim/N_hover * 1/1000 * eff_B_h * eff_PE_h * eff_M_h / 1.2

    # Calculate available rpm based on allowable Mach at the blade tips
    rpm_max_hover = M_t_max*a*60 / (np.pi * 2*r_out)
    # Cruise rpm is hpver rpm times the ratio of cruise vs hover power
    rpm_cr = rpm_max_hover * (P_br_cruise/P_br_hover)**(2/3)

    print("For config 2, the maximum allowable rpm in hover are:", rpm_max_hover, "[rpm]")
    print("For config 2, the maximum allowable rpm in cruise are:", rpm_cr, "[rpm]")

    # Mach speed at blade tip during cruise based on cruise rpm
    M_t_cr = np.pi*2*r_out*rpm_cr / (a*60)

    # The 1s are number of propellers (we calculate individually each engine, so 1)
    noise = brr.Noise(P_br_cruise, P_br_hover, 2*r_out, num_blades, 1, 1, rpm_max_hover, rpm_cr, a, M_t_h=M_t_max, M_t_cr=M_t_cr)

    print("Noise level in cruise per engine:", noise.SPL_cr(), "[dB]")
    print("Noise level in cruise, total:", brr.sum_noise(np.ones((1, N_cruise))[0] * noise.SPL_cr()), "[dB]")
    print("")

    print("Noise level in hover per engine:", noise.SPL_hover(), "[dB]")
    print("Noise level in hover, total:", brr.sum_noise(np.ones((1, N_hover))[0] * noise.SPL_hover()), "[dB]")
    print("")

elif Prop_config == 3:
    M_t_max = 0.7

    r_out_wing_eng = ((xc_wing_eng_end-xc_wing_eng_start)*c_r/2 - (xc_wing_eng_end-xc_wing_eng_start)*(1-taper)*c_r*xb_wing_eng_start/2) / \
                     (1 + 2*(xc_wing_eng_end-xc_wing_eng_start)*(1-taper)/b**2)

    wing_prop_hub_ratio = 0.2
    area_wing_prop = np.pi * (r_out_wing_eng**2 - (wing_prop_hub_ratio*r_out_wing_eng)**2)
    area_tilt_eng = (disk.A - 2*area_wing_prop)/4
    r_out = np.sqrt(area_tilt_eng / (np.pi * (1 - D_inner_ratio**2)))

    area_ratio_tilt = 4*area_tilt_eng/disk.A
    mass_tilt_eng = area_ratio_tilt*MTOM
    mass_wing_eng = MTOM-mass_tilt_eng

    P_cr_tilt = prop.PropulsionCruise(mass_tilt_eng, N_cruise, area_tilt_eng, eff_P_cr, eff_D_cr, eff_F_cr, eff_M_cr,
                                      eff_PE_cr, eff_B_cr, rho, V_cruise, MTOW/LD_ratio)

    P_h_tilt = prop.PropulsionHover(mass_tilt_eng, 4, area_tilt_eng, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                    disk.v_e_hover(), 0, rho, Ducted)
    P_h_wing = prop.PropulsionHover(mass_wing_eng, 2, area_wing_prop, eff_D_h, eff_F_h, eff_M_h, eff_PE_h, eff_B_h,
                                    disk.v_e_hover(), 0, rho, Ducted)

    # Calculate available rpm based on allowable Mach at the blade tips
    rpm_max_tilt = M_t_max*a*60 / (np.pi*2*r_out)
    rpm_max_wing = M_t_max*a*60 / (np.pi*2*r_out_wing_eng)
    print("For config 3, the maximum allowable rpm for the tilting engines are:", rpm_max_tilt, "[rpm]")
    print("For config 3, the maximum allowable rpm for the wing engines are:", rpm_max_wing, "[rpm]")
    print("")

    # Shaft power
    P_h_wing = P_h_wing.P_hover() * 1/1000 * eff_B_h * eff_PE_h * eff_M_h / 1.2
    P_h_tilt = P_h_tilt.P_hover() * 1/1000 * eff_B_h * eff_PE_h * eff_M_h / 1.2

    P_cr = P_cr_tilt.P_cr() * 1/1000 * eff_B_cr * eff_PE_cr * eff_M_cr / 1.2

    # The 1s are number of propellers (we calculate individually each engine, so 1)
    noise_tilt = brr.Noise(P_cr, P_h_tilt, 2*r_out, num_blades, 1, 1, rpm_max_tilt, rpm_max_tilt, a, M_t_h=M_t_max, M_t_cr=M_t_max)
    noise_wing = brr.Noise(0, P_h_wing, 2*r_out, num_blades, 1, 1, rpm_max_wing, rpm_max_wing, a, M_t_h=M_t_max, M_t_cr=M_t_max)

    print("Noise level in cruise per tilting engine:", noise_tilt.SPL_cr(), "[dB]")
    print("Noise level in cruise, total:", brr.sum_noise(np.ones((1, N_cruise))[0] * noise_tilt.SPL_cr()), "[dB]")
    print("")

    print("Noise level in hover per tilting engine:", noise_tilt.SPL_hover(), "[dB]")
    print("Noise level in hover per wing engine:", noise_wing.SPL_hover(), "[dB]")
    print("Noise level in hover, total:", brr.sum_noise([noise_tilt.SPL_hover(), noise_tilt.SPL_hover(),
                                                         noise_tilt.SPL_hover(), noise_tilt.SPL_hover(),
                                                         noise_wing.SPL_hover(), noise_wing.SPL_hover()]), "[dB]")
    print("")


# # ----- Sensitivity study of the noise formulas -----
# # Shaft power of each engine in kW
# P_br_cruise = P_cr_estim/N_cruise * 1 / 1000 * 0.6
# P_br_hover = P_hover_estim/N_hover * 1 / 1000 * 0.6
#
# # The 1s are number of propellers (we calculate individually each engine, so 1)
# noise = brr.Noise(P_br_cruise, P_br_hover, 2*r_out, num_blades, 1, 1, rpm, a)
#
#
# print("Noise level in cruise per engine:", noise.SPL_cr(), "[dB]")
# print("Noise level in cruise, total:", brr.sum_noise(np.ones((1, N_cruise))[0] * noise.SPL_cr()), "[dB]")

# print("Noise level in hover:", noise.SPL_hover())

# print("Speed of sound:", a)
# print("Outer radius:", r_out)
