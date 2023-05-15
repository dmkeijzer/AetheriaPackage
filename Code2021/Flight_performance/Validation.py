import numpy as np
import matplotlib.pyplot as plt
from Aero_tools import speeds, ISA
from Flight_performance_final import evtol_performance, mission
from constants import g
import scipy.optimize as optimize
from Preliminary_Lift.main_aero import Cl_alpha_curve, CD_a_w, CD_a_f, alpha_lst, Drag
from Final_optimization import constants_final as const

class validation:
    def __init__(self, mass, cruising_alt, cruise_speed, CL_max, P_max, wing_surface, A_disk, t_loiter = 30*60,
                 rotational_rate = 5, roc = 5, rod = 5, mission_dist = 300e3):

        # Initial values for all the input parameters
        self.mass = mass
        self.h_cr = cruising_alt
        self.v_cr = cruise_speed
        self.S    = wing_surface
        self.CL_max = CL_max
        self.t_lt = t_loiter
        self.rot  = rotational_rate
        self.roc  = roc
        self.rod  = rod
        self.dist = mission_dist
        self.A_disk = A_disk
        self.P_max = P_max

        # Range over which the values are allowed to vary
        var     = 0.05    # [-] +- percentage variation
        N_pts   = 5       # [-] Number of different variations per parameter

    def monte_carlo_sim(self, var, N_sim):

        max_var = 1 + var
        min_var = 1 - var
        E = np.zeros(N_sim)
        t = np.zeros(N_sim)

        for i in range(N_sim):

            print('Progress:', np.round(100*i/N_sim), '%')

            m = mission(mass         = np.random.default_rng().uniform(low = self.mass*min_var, high = self.mass*max_var),
                        cruising_alt = self.h_cr,
                        cruise_speed = self.v_cr,
                        CL_max       = np.random.default_rng().uniform(low = self.CL_max*(1-2*var), high = self.CL_max),
                        wing_surface = np.random.default_rng().uniform(low = self.S*min_var, high = self.S*max_var),
                        A_disk       = np.random.default_rng().uniform(low = self.A_disk*min_var, high = self.A_disk*max_var),
                        P_max        = np.random.default_rng().uniform(low = self.P_max*min_var, high = self.P_max*max_var),
                        Cl_alpha_curve = Cl_alpha_curve,
                        CD_a_w = CD_a_w,
                        CD_a_f = CD_a_f,
                        alpha_lst = alpha_lst,
                        Drag = Drag)

            E[i], t[i],_,_,_ = m.total_energy()




        plt.hist(E)
        plt.show()
        np.savetxt("../Flight_performance/Saved_data/validation_energy", E)
        np.savetxt("../Flight_performance/Saved_data/validation_time", t)

    #
    # def energy_bounds_takeoff(self):
    #
    #     # Lower bound, based on the optimal take-off trajectory found by Chauhan
    #     lb = mission(mass = 725, cruising_alt = 305, cruise_speed = 67, CL_max = 1.2, wing_surface = 9, A_disk = 14.14,
    #                  t_loiter = 30*60, rotational_rate = 1, plotting= True, roc = 15, P_max = self.P_max)
    #
    #     # Get the lower bound of the energy required to climb
    #     d_lb, E_lb, t_lb = lb.numerical_simulation(0.01, 0, np.pi/2, 305, 67)
    #
    #     # Energy calculated by Chauhan in Joules
    #     E_ch = 1871*3600
    #
    #     print('Comparing the lower bound: ', 100*(E_ch - E_lb)/E_lb, '%', E_ch, E_lb)
    #
    #     self.monte_carlo_sim(0.15, 50)
    #
    #     # Upper bound
    #     ub = mission(mass=3175, cruising_alt=3000, cruise_speed=83, CL_max=1.2, wing_surface=8.464, A_disk=2.052,
    #                  t_loiter=30 * 60, rotational_rate=1, plotting=True, roc = 7.2, P_max = self.P_max)
    #
    #     d_ub, E_ub, t_ub = ub.numerical_simulation(0.01, 0, np.pi / 2, 3000, 83)
    #
    #     # Energy from Nathen
    #     E_nt = 511000*451 + 12*1421000 + 15*224000
    #
    #     print('Comparing the upper bound: ', 100*(E_nt - E_ub)/E_ub, E_nt, E_ub)

mass = 2800
cruising_alt = 1000
cruise_speed = 72
CL_max = 1.5856
wing_surface = 19.82
EOM = mass - (const.m_pax*4 + const.m_cargo_tot)
A_disk = 0.795*const.n_prop
P_max  = 1.81e6

validate = validation(mass = mass, cruising_alt = cruising_alt, cruise_speed = cruise_speed,
                      CL_max = 1.7, wing_surface = 14, A_disk = 8, P_max = P_max)
#validate.energy_bounds_takeoff()
validate.monte_carlo_sim(0.10, 100)

