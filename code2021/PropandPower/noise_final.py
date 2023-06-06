import numpy as np
import code2021.Final_optimization.Aero_tools as at
import Midterm_files.Noise as noise_midterm

"""
Program to calculate the noise of propellers


Method obtained from:
A Review of Aerodynamic Noise From Propellers, Rotors, and Lift Fans 
Jack E. Made 
Donald W. Kurtz
"""
# Atmospherical parameters
ISA = at.ISA(1000)
rho = ISA.density()
a = ISA.soundspeed()

g0 = 9.80665
MTOM = 3024.8012022968796

# Propeller parameters
B = 6
P_br = 13627.72  # [W]
P_br_hp = 18.27  # [hp]

R = 0.5029297
D = 2*R
D_ft = D / 0.3048
dimensionless_distance = 1/D_ft
print("Dimensionless distance:", dimensionless_distance)

xi_0 = 0.1

area_prop = np.pi * (R**2 - (xi_0*R)**2)
A_tot = area_prop*12

V_cruise = 72.18676185339652

rpm_cr = 1090
Omega_cr = rpm_cr * 2 * np.pi / 60
n_cr = Omega_cr / (2 * np.pi)

V_tip = Omega_cr*R
M_tip = V_tip/a
print("Tip Mach number:", M_tip)


# Init
Noise_cruise = 0

# Step 1, need power per engine in hp, and get base level from graph. For 18 hp it is approx 107 dB
L1 = 107
Noise_cruise += L1

# Step 2
# Add 2O log 4/B, where B is the number of blades; and add 40 log 15.5/D
Noise_cruise += 20*np.log10(4/B) + 40 * np.log10(15.5/D_ft)

# Correction, use dimensionless distance (0.3030) and tip Mach (0.21)
correction_tip_Mach = -20

Noise_cruise += correction_tip_Mach

# Correction directional factor
correction_direction = 4  # +4 dB is the average maximum

Noise_cruise += correction_direction

# Subtract 20 log (T - l), where r is the distance, in ft, from the center of the propeller.
# First check at 1 m
Noise_cruise_1m = Noise_cruise - 20*np.log10(1/0.3048 - 1)

# Distance of interest
r = 1000/0.3048  # Approx 100 m

Noise_cruise -= 20*np.log10(r - 1)


print("The propeller noise at cruise at", r*0.3048, "m from the propeller is", Noise_cruise, "dB")

print("The propeller noise at cruise at 1 m from the propeller is", Noise_cruise_1m, "dB")
print("")

# TODO: sum noise of the 12 propellers
def sum_noise(noises):
    tens = np.ones((np.shape(noises))) * 10
    summed_noise = 10 * np.log10(np.sum(np.power(tens, np.array(noises)/10)))
    return summed_noise


print("Noise level in cruise at", r*0.3048, "m from the propeller, total:", sum_noise(np.ones((1, 12))[0] * Noise_cruise), "[dB]")
print("")

"""
Noise midterm
"""
T_max = 34311.7687171136
P_max = T_max**(3/2) / np.sqrt(2*rho*A_tot)

T_h = MTOM*g0
P_h = T_h**(3/2) / np.sqrt(2*rho*A_tot)


rpm_h = 4000
Omega_h = rpm_h * 2 * np.pi / 60

V_tip_h = Omega_h*R
M_tip_h = V_tip_h/a

midterm_noise = noise_midterm.Noise(P_br/1000, P_h/12, D, B, 12, 12, rpm_h, rpm_cr, a, M_tip_h, M_tip)

print("The propeller noise in cruise at 1 metre from the propeller is", midterm_noise.SPL_cr())
print("The propeller noise in hover at 1 metre from the propeller is", midterm_noise.SPL_hover())
