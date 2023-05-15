import integration_class_ar_input as int_class
import constants_final as const
import sys
import os

sys.path.append(r"C:\Users\damie\OneDrive\Desktop\Damien\Honours\Wigeon_code\DSE2021\Preliminary_Lift\Airfoil_data")
os.chdir(r"C:\Users\damie\OneDrive\Desktop\Damien\Honours\Wigeon_code\DSE2021\Final_optimization")

# Initial estimates
MTOM = 2800
V_cr = 66
h_cr = 1000
C_L_cr = 0.8
CLmax = 1.68
prop_radius = 0.55
de_da = 0.25
Sv = 1.1
V_stall = 40
max_power = 1477811
AR_wing1 = 9.511387725171947
AR_wing2 = 9.978405107509532
Sr_Sf = 1.2
s1 = (1 + Sr_Sf)**-1

# Positions of the wings [horizontally, vertically]
xf = 0.5
zf = 0.3
xr = 7.38451

zr = 1.7
max_thrust_stall = MTOM*const.g*0.1
# Initial estimates for the variables
initial_estimate = [MTOM, 0, V_cr, h_cr, C_L_cr, CLmax, prop_radius, de_da, Sv, V_stall, max_power, AR_wing1,
                    AR_wing2, Sr_Sf, s1, xf, zf, xr, zr, max_thrust_stall, 1, 1.5, 2.4, 2.6, 8, 2.490737144567832]

# Optimisation class
optimisation_class = int_class.RunDSE(initial_estimate)

# Run the file for # iterations
N_iter = 10
optim_outputs, internal_inputs, other_outputs = optimisation_class.multirun(N_iter, optim_inputs=[])

print(optim_outputs)
print("")
print(internal_inputs)
print("")
print(internal_inputs)

