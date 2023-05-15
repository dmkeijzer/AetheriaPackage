from Flight_performance_final import mission, evtol_performance
from Aero_tools import speeds
from Preliminary_Lift.main_aero import Cl_alpha_curve, CD_a_w, CD_a_f, alpha_lst, Drag
import Final_optimization.constants_final as const

from Preliminary_Lift.Drag import componentdrag



# ========== Inputs ==========
mass = 2800
cruising_alt = 1000
cruise_speed = 72
CL_max = 1.5856
wing_surface = 19.82
EOM = mass - (const.m_pax*4 + const.m_cargo_tot)
A_disk = 0.795*const.n_prop
P_max  = 1.81e6

# Energy estimation and plotting
mission_profile = mission(mass, cruising_alt, cruise_speed, CL_max, wing_surface, A_disk, P_max,
                          Cl_alpha_curve, CD_a_w, CD_a_f, alpha_lst, Drag, t_loiter = 15*60,
                          plotting = True)

E_tot, t_tot, max_power, max_thrust,_ = mission_profile.total_energy()

print('Total energy', E_tot/1e6, 'MJ')
print("Total time", t_tot/3600, 'hr')
print("Max power", max_power/1e3, 'kW')

# Other performance estimates
performance = evtol_performance(cruising_alt, cruise_speed, wing_surface, CL_max, mass, 880e6,
                                EOM, 15*60, A_disk, P_max,
                                Cl_alpha_curve, CD_a_w, CD_a_f, alpha_lst, Drag)

# Performance things
R,_ = performance.range(1000, 72, mass = mass, loiter=True)
print('range', R)
performance.power_polar(cruising_alt)
V_climb = performance.climb_performance()
performance.vertical_climb()
performance.payload_range()

# Optimal speeds
V = speeds(cruising_alt, mass, CL_max, wing_surface, Drag)

print("===== Optimal speeds =====")
print("Stall speed:", V.stall())
print("Cruise speed:", V.cruise()[0])
print("Climb speed:", V_climb, V.climb())
