import AetheriaPackage.GeneralConstants as const
from AetheriaPackage.GeneralConstants import *
from AetheriaPackage.data_structs import *
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy.optimize import minimize


def compute_tank_radius(V, n, l_tank):
    """_summary_

    :param V: volume of tank
    :param n: number of tanks
    :param l_tank:  length of tank
    :return: returns radius of tank
    """    

    roots = np.roots([-2*np.pi / 3, np.pi * l_tank, 0, -V/n]) # Find positive roots of cubic function of Tank Volume (two tanks)
    positive_roots = [root.real for root in roots if np.isreal(root) and root > 0]
    # r = positive_roots[0] # radius of tank
    r = np.min(roots[roots > 0]) # Select the correct root of the equation
    return r
"""CALCULATE TAIL LENGTH BASED ON BETA AND ASPECT RATIO"""
def find_tail_length(h0, b0, Beta, V, l, AR, n):
    roots = np.roots([-2*np.pi / 3, np.pi * l, 0, -V/n]) # Find positive roots of cubic function of Tank Volume (two tanks)
    positive_roots = [root.real for root in roots if np.isreal(root) and root > 0]
    # r = positive_roots[0] # radius of tank
    r = np.min(roots[roots > 0]) # Select the correct root of the equation
    bc = 2 * n * r # width of crashed fuselage at end of tank
    hc = bc / AR # height of crashed fuselage at end of tank
    A_f = bc ** 2 / (AR * Beta ** 2) # area of fuselage at end of tank
    hf = np.sqrt(A_f / AR) # height of fuselage at end of tank
    bf = A_f/hf # width of fuselage at end of tank
    l_t = h0 * l / (h0 - hf) # length of tail
    upsweep = np.arctan2((h0 - hf), l) # upsweep angle
    return l_t, upsweep, bc, hc, hf, bf

"""CONVERGE TAIL LENGTH BY CONVERGING ASPECT RATIO"""
def converge_tail_length(h0, b0, Beta, V, l, ARe, n):
    AR0 = b0/h0
    AR = AR0
    error, i = 1, 0 # iteration error and number
    ARarr = [] # aspect ratio array
    while error > 0.005: # stop when error is smaller than 0.5%
        ARarr.append(AR)
        tail_data = list(find_tail_length(h0, b0, Beta, V, l, AR, n))
        AR = l / tail_data[0] * (ARe - AR0) + AR0
        error = np.abs((ARarr[-1] - AR)/AR)
        i += 1
        if i > 200: # stop if iteration number if more than 200 (no convergence)
            error = 0
    #print("Converged after: ", i, "iterations to AR: ", AR)
    tail_data.append(AR)
    return tail_data # returns tail length, upsweep, bc, hc, hf, bf

"""MAKE 2D SENSITIVY PLOT FOR BETA AND ARe"""
def plot_variable(h0, b0, V, l_tank, n,  parameter, parameter_values, fixed_parameter, fixed_value):
    l_tail = []

    for i in range(len(l_tank)):
        l_tail_row = []
        for j in range(len(parameter_values)):
            if parameter == 'ARe':
                ARe = parameter_values[j]
                Beta = fixed_value
            elif parameter == 'Beta':
                ARe = fixed_value
                Beta = parameter_values[j]

            l_t, upsweep, bc, hc, hf, bf, AR = converge_tail_length(h0, b0, Beta, V, l_tank[i], ARe, n)

            if 1 <= l_t <= 7.5 and hf < h0 and l_t > l_tank[i] and bf < b0  and AR > 0 and bc > 0 and hf > 0 and bf > 0 and hc > 0 and hc>bc/n:
                #and hc > bc / n
                l_tail_row.append(l_t)
            else:
                l_tail_row.append(np.nan)

        l_tail.append(l_tail_row)

    l_tank, parameter_values = np.meshgrid(l_tank, parameter_values)
    l_tail = np.array(l_tail)

    # Plot the colored plot
    fig, ax = plt.subplots()

    if parameter == 'ARe':
        cmap = 'viridis_r'
        start_value = 3  # Set the desired start value

    elif parameter == 'Beta':
        cmap = 'viridis_r'
        start_value = 2.5  # Set the desired start value

    levels = np.arange(start_value, np.nanmax(l_tail) + 0.51, 0.5)  # Set the colorbar range to exactly 0.4

    norm = BoundaryNorm(levels, ncolors=plt.cm.get_cmap(cmap).N)

    c = ax.contourf(l_tank, parameter_values, l_tail.T, levels=levels, cmap=cmap, norm=norm)

    ax.set_xlabel('Tank Length [m]')
    ax.set_ylabel(parameter)
    ax.set_title(f'Fixed {fixed_parameter}: {fixed_value}')

    # Add colorbar
    cbar = plt.colorbar(c, label='Tail Length [m]')
    cbar.set_ticks(levels)
    cbar.set_ticklabels([f'{tick:.1f}' for tick in levels])  # Format tick labels with two decimal places

    plt.savefig(os.path.join(os.path.expanduser("~"), "Downloads", "sensitivity_tail_" + parameter + "_" + str(fixed_value) +  ".pdf"), bbox_inches= "tight")
    plt.show()

def minimum_tail_length(h0, b0, Beta, V, l_tank, ARe, n, plot= False):
    l_tail = []
    l_tank = list(l_tank)
    indices_to_remove = []  # Track indices to be removed

    for i in range(len(l_tank)):
        l_t, upsweep, bc, hc, hf, bf, AR = converge_tail_length(h0, b0, Beta, V, l_tank[i], ARe, n)
        l_tail.append(l_t)

        if l_t < l_tank[i] or l_t > 8 or AR < 0 or bf < 0 or hf < 0 or hc < 0 or bc < 0 or hc<bc/n or bf>b0 or hf>h0:
            indices_to_remove.append(i)

    if plot:
        plt.plot(l_tank, l_tail, label= "tail length")
        plt.xlabel("Tank length")
        plt.ylabel("Tail length")
        plt.grid()
        plt.legend()
        plt.show()

    # Remove values from l_tail based on indices to remove
    l_tail = [l for i, l in enumerate(l_tail) if i not in indices_to_remove]

    # Remove values from l_tank based on indices to remove
    l_tank = [l for i, l in enumerate(l_tank) if i not in indices_to_remove]

    if plot:
        plt.plot(l_tail, l_tank)
        plt.xlabel("Tail length [m]")
        plt.ylabel("Tank length [m]")
        plt.show()

    l_tail = np.array(l_tail)
    if len(l_tail) > 0:
        min_index = np.argmin(l_tail)
        tail_data = converge_tail_length(h0, b0, Beta, V, l_tank[min_index], ARe, n)
        tail_data.append(l_tank[min_index])
    else:
        raise Exception("No possbile tail length")


    return tail_data

def stress_strain_curve(stress_peak, strain_peak, plain_stress, e_d):
    E = stress_peak/strain_peak
    e = np.arange(0,1,0.001)
    densification = e_d
    strain_p = stress_peak / E
    crush_strength = np.zeros(len(e))

    for i in range(len(e)):
        if e[i] <= strain_p:
            crush_strength[i] = e[i]*E
        if e[i] > strain_p and e[i] <= densification:
            crush_strength[i] = plain_stress
        if e[i] > densification:
            crush_strength[i] = plain_stress + (e[i]-densification)*E

    #plt.plot(e, crush_strength/(10**6))
    #plt.show()
    return crush_strength, e


def decel_calculation(s_p, s_y, e_0, v0, e_d, height,m, PRINT=False):
    a, v, s, t  = [0], [v0], [0], [0]
    Ek = [0.5*m*v[-1]**2]
    s_tot = height
    dt = 0.000001
    g = 9.81
    A = m*20*g/(s_p)
    Strain = [0]
    Strainrate = [0]

    sigma_cr, e = stress_strain_curve(s_y, e_0, s_p, e_d)

    while v[-1] > 0:
        strain = s[-1]/s_tot
        index = np.abs(e - strain).argmin()
        Fcrush = sigma_cr[index]*A
        #Fcrush = 0.315*10**6
        ds = v[-1]*dt
        work_done = Fcrush*ds
        Ek.append(Ek[-1]- work_done)

        if Ek[-1] > 0:
            v.append(np.sqrt(2*Ek[-1]/m))
            a.append((v[-1]-v[-2])/dt)
            t.append(t[-1]+dt)
            s.append(s[-1] + ds)
            Strain.append(strain)
            Strainrate.append((Strain[-1]-Strain[-2])/dt)
        else:
            Ek.pop(-1)
            break
    if PRINT:
        plt.plot(t[1:-1], np.array(Strain[1:-1]))
        plt.xlabel("Time")
        plt.ylabel("Strain")
        plt.show()

        plt.plot(t[1:-1], np.array(Strainrate[1:-1]))
        plt.xlabel("Time")
        plt.ylabel("Strain rate")
        plt.show()

        plt.plot(t[1:-1], np.array(a[1:-1])/g)
        plt.xlabel("Time")
        plt.ylabel("Acceleration")
        plt.show()

        plt.plot(t, v)
        plt.xlabel("Time")
        plt.ylabel("Velocity")
        plt.show()

        plt.plot(t, s)
        plt.plot([0, t[-1]],[e_d*s_tot, e_d*s_tot])
        plt.xlabel("Time")
        plt.ylabel("Distance")
        plt.show()
    max_g = min(np.array(a))

    return s[-1], A, max_g

def simple_crash_box(m, a, sigma_cr, v):
    s = v**2/(2*a)
    A = m*a/sigma_cr
    print(s, A)
    return s, A

def crash_box_height_convergerence(plateau_stress, yield_stress, e_0, e_d, v0, s0, m):
    I, s_arr, i = [], [], 0
    height = s0
    error = 1
    while error > 0.0005:
        travel_distance, A, max_g = decel_calculation(plateau_stress, yield_stress, e_0, v0, e_d, height, m, False)
        new_height = travel_distance/e_d
        error = abs((new_height-height)/height)
        height = new_height
        I.append(i)
        s_arr.append(height)
        i += 1
    #print(height, A, max_g/9.81)
    #plt.plot(I, s_arr)
    #plt.show()
    return travel_distance, A




def get_fuselage_sizing(h2tank, fuelcell, perf_par,fuselage, validate= False):

    crash_box_height, crash_box_area = crash_box_height_convergerence(const.s_p, const.s_y, const.e_0, const.e_d, const.v0, const.s0, perf_par.MTOM)
    fuselage.height_fuselage_inner = fuselage.height_cabin + crash_box_height
    fuselage.height_fuselage_outer = fuselage.height_fuselage_inner + const.fuselage_margin

    fuselage.volume_powersys = h2tank.volume(perf_par.mission_energy)
    if validate:
        print(f"|{fuselage.volume_powersys=:^20.4e}|")
    # l_tail, upsweep, bc, hc, hf, bf, AR, l_tank = minimum_tail_length(fuselage.height_fuselage_inner, fuselage.width_fuselage_inner, const.beta_crash, h2tank.volume(perf_par.energyRequired/3.6e6) ,np.linspace(1, 7, 40), const.ARe, const.n_tanks)
    l_tail, upsweep, bc, hc, hf, bf, AR, l_tank = minimum_tail_length(fuselage.height_fuselage_inner, fuselage.width_fuselage_inner, const.beta_crash, fuselage.volume_powersys ,np.linspace(1, 9, 100), const.ARe, const.n_tanks, plot= validate)
    radius = compute_tank_radius(fuselage.volume_powersys, 2, l_tank)

    fuselage.length_tail = l_tail
    fuselage.length_tank = l_tank
    fuselage.tank_radius = radius
    fuselage.upsweep = upsweep 
    fuselage.bc = bc
    fuselage.crash_box_area =  crash_box_area
    fuselage.hc = hc
    fuselage.bf = bf
    fuselage.hf = hf
    fuselage.limit_fuselage = fuselage.length_cockpit + fuselage.length_cabin + l_tail + fuelcell.depth + const.fuselage_margin 
    fuselage.length_fuselage = fuselage.length_cockpit + fuselage.length_cabin + l_tail + fuelcell.depth + const.fuselage_margin 

    return fuselage

class PylonSizing():
    def __init__(self, engine, L):
        self.mass_eng = engine.mass_pertotalengine
        self.L = L
        self.Tmax =  2.5*2200*9.81/6
        self.moment = self.Tmax*L

    def I_xx(self, x): return np.pi/4 *  ((x[0] + x[1])**4 - x[0]**4)

    def get_area(self, x):
        return np.pi*((x[1] + x[0])**2 - x[0]**2)

    def weight_func(self, x):
        return np.pi*((x[1] + x[0])**2 - x[0]**2)*const.rho_composite*self.L


    def get_stress(self, x):
        return (self.moment*(x[1] + x[0]))/self.I_xx(x)

    # def r2_larger_than_r1(self, x):
    #     # print(f"r2>r1 = {x[1] - x[0]}")
    #     return x[1] - x[0]

    def column_buckling_constraint(self, x):
        # print(f"r1, r2 = {x[0], x[1]}")
        # print(f"column buckling = {(np.pi**2*const.E_alu*self.I_xx(x))/(self.L**2*self.get_area(x))- self.get_stress(x)}")
        return (np.pi**2*const.E_composite*self.I_xx(x))/(self.L**2*self.get_area(x)) - self.get_stress(x)

    def von_mises_constraint(self, x):
        # print(f"Von Mises = {const.sigma_yield -1/np.sqrt(2)*self.get_stress(x)} ")
        return const.sigma_yield - 1/np.sqrt(2)*self.get_stress(x)

    def eigenfreq_constraint(self, x):
        # print(f"Eigenfrequency = {1/(2*np.pi)*np.sqrt((3*const.E_alu*self.I_xx(x))/(self.L**3*self.mass_eng))}")
        print(f"Ixx = {self.I_xx(x)}")
        return 1/(2*np.pi)*np.sqrt((3*const.E_composite*self.I_xx(x))/(self.L**3*self.mass_eng)) - const.eigenfrequency_lim_pylon


    def  optimize_pylon_sizing(self, x0):

        cons = (
            {'type': 'ineq', 'fun': self.column_buckling_constraint },
                {'type': 'ineq', 'fun': self.von_mises_constraint }
                # {'type': 'ineq', 'fun': self.eigenfreq_constraint}
                )
        bnds = ((0.095, 0.1), (0.001,0.2))

        res = minimize(self.weight_func, x0, method='SLSQP', bounds=bnds, constraints=cons)

        return res


#Moments of Inertia
def i_xx_solid(width,height):
    return width*height*height*height/12
def i_yy_solid(width,height):
    return width*width*width*height/12
def j_z_solid(width,height):
    return width*height*(width*width + height*height)/12

def i_xx_thinwalled(width,height,thickness):
    return 1/3 * width*height*height*thickness
def i_yy_thinwalled(width,height,thickness):
    return 1/3 * width*width*height*thickness
def j_z_thinwalled(width,height,thickness):
    return (height+width)*height*width*thickness/3


"""NORMAL STRESS"""
def bending_stress(moment_x,moment_y,i_xx,i_yy,i_xy,x,y):
    return((moment_x*i_yy-moment_y*i_xy)*y + (moment_y*i_xx - moment_x*i_xy)*x)/(i_xx*i_yy-i_xy*i_xy)
def normal_stress(force,area):
    return force/area
    

"""SHEAR STRESS"""
def torsion_circular(torque,dens,j_z):
    return torque*dens/j_z

def torsion_thinwalled_closed(torque,thickness,area):
    return torque/(2*thickness*area)

def maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, pos=True):
    n = lambda CL, V, WoS: 0.5 * const.rho_cr* V ** 2 * CL / WoS
    Vc, VD = Vs
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return min(n(CLmax, V, WoS), nmax) if pos else \
    ( max(-n(CLmax, V, WoS), nmin) if V <= Vc else interpolate(V, Vc, VD, nmin, 0))

def plotmaneuvrenv(WoS, Vc, CLmax, nmin, nmax):
    VD = 1.2*Vc
    Vs = Vc, VD
    x = np.linspace(0, VD, 100)
    ax = sns.lineplot(x=x, y=[maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, True) for V in x], color='blue', zorder=3)
    sns.lineplot(x=x, y=[maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, False) for V in x], color='blue', label='Manoeuvre Envelope',zorder=3)
    ax.set(xlabel="V [m/s]", ylabel="n [-]")
    plt.plot([VD, VD], [maneuvrenv(VD, Vs, WoS, CLmax, nmin, nmax, True), maneuvrenv(VD, Vs, WoS, CLmax, nmin, nmax, True)], color='blue',zorder=3)
    plt.plot([VD, VD],[0, nmax], color='blue',zorder=3)
    plt.grid(True)
    plt.xlim(0,VD+7)
    plt.plot([-5,VD+7],[0,0], color='black', lw=1)


    #plt.savefig('manoeuvre_env.png')
    return np.max([maneuvrenv(V, Vs, WoS, CLmax, nmin, nmax, True) for V in x])

def posgustload(V, Vs, us, ns, CLalpha, WoS):
    n = lambda V, u: 1 + const.rho_cr * V * CLalpha * u / (2 * WoS)
    (ub, uc, ud), (Vb, Vc, VD), (nb, nc, nd)  = us, Vs, ns
    interpolate = lambda V, V1, V2, n1, n2: n1 + (V - V1) * (n2 - n1) / (V2 - V1)
    return n(V, ub) if 0 <= V <= Vb else \
    ( interpolate(V, Vb, Vc, nb, nc) if Vb < V <= Vc else interpolate(V, Vc, VD, nc, nd) )

neggustload = lambda V, Vs, us, ns, CLalpha, WoS: 2 - posgustload(V, Vs, us, ns, CLalpha, WoS)


def plot_dash(V, n):
    plt.plot([0, V],[1, n], linestyle='dashed', color='black', zorder=1, alpha=0.5)


def plotgustenv(V_s, Vc, CLalpha, WoS, TEXT=False):
    n = lambda V, u: 1 + const.rho_cr * V * CLalpha * u / (2 * WoS)
    #Vb = np.sqrt(n(Vc, uc))*V_s
    Vb = Vc - 22.12
    Vb, Vc, VD = Vs = (Vb, Vc, 1.2*Vc) # Change if VD Changes
    us = const.ub, uc, ud  # Obtained from CS
    nb, nc, nd = ns = n(Vb, ub), n(Vc, uc), n(VD, ud)
    x = np.linspace(0, VD, 100)
    ax = sns.lineplot(x=x, y=[posgustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='black', zorder=2)
    ax.set(xlabel="V [m/s]", ylabel="n [-]")
    sns.lineplot(x=x, y=[neggustload(V, Vs, us, ns, CLalpha, WoS) for V in x], color='black', label='Gust Load Envelope',zorder=2)
    plt.plot([VD, VD], [neggustload(VD, Vs, us, ns, CLalpha, WoS), posgustload(VD, Vs, us, ns, CLalpha, WoS)], color='black',zorder=2)
    plot_dash(Vc, nc)
    plot_dash(Vc, 2 - nc)
    plot_dash(VD, nd)
    plot_dash(VD, 2 - nd)
    plt.plot([Vb, Vb], [2-nb, nb], linestyle='dashed', color='black', zorder=1, alpha=0.5)
    plt.plot([Vc, Vc], [2-nc, nc], linestyle='dashed', color='black', zorder=1, alpha=0.5)
    if TEXT:
        plt.text(Vb + 1, 0.1, 'Vb', fontsize = 11, weight='bold')
        plt.text(V_s + 1, 0.1, 'Vs', fontsize=11, weight='bold')
        plt.text(Vc + 1, 0.1, 'Vc', fontsize=11, weight='bold')
        plt.text(VD + 1, 0.1, 'Vd', fontsize=11, weight='bold')
        plt.plot([V_s,V_s],[0, 0.05], color='black')
        plt.plot([Vb, Vb], [0, 0.05], color='black')
        plt.plot([Vc, Vc], [0, 0.05], color='black')
        plt.plot([Vc, Vc], [0, 0.05], color='black')



    return np.max([posgustload(V, Vs, us, ns, CLalpha, WoS) for V in x])
    # plt.savefig('gust.png')

def get_gust_manoeuvr_loadings(perf_par, aero):

    nm = plotmaneuvrenv(perf_par.wing_loading_cruise, const.v_cr, aero.cL_max, const.n_min_req, const.n_max_req)
    ng = plotgustenv(const.v_stall, const.v_cr , aero.cL_alpha, perf_par.wing_loading_cruise, TEXT=False)

    perf_par.n_max, perf_par.n_ult = max(nm, ng), max(nm, ng)*1.5

    return perf_par

class VtolWeightEstimation:
    def __init__(self) -> None:
        self.components = []

    def add_component(self, CompObject):
        """ Method for adding a component to the VTOL

        :param CompObject: The component to be added to the VTOL
        :type CompObject: Component parent class
        """        
        self.components.append(CompObject)  

    def compute_mass(self):
        """ Computes the mass of entire vtol

        :return: Entire mass of VTOL
        :rtype: float
        """        
        mass_lst = [i.return_mass() for i in self.components]
        return np.sum(mass_lst)*const.oem_cont

class Component():
    """ This is the parent class for all weight components, it initalized the mass
    attribute and a way of easily returning it. This is used in VtolWeightEstimation.
    """    
    def __init__(self) -> None:
        self.mass = None

    def return_mass(self): return self.mass


class WingWeight(Component):
    def __init__(self, mtom, S, n_ult, A):
        """Retunrs the weight of the wing, Cessna method cantilever wings pg. 67 pt 5. Component weight estimation Roskam

        :param mtom: maximum take off mass
        :type mtom: float
        :param S: Wing area
        :type S: float
        :param n_ult: Ultimate load factor
        :type n_ult: float
        :param A: Aspect ratio
        :type A: float
        """        
        super().__init__()
        self.id = "wing"
        self.S_ft = S*10.7639104
        self.n_ult = n_ult
        self.A = A
        self.mtow_lbs = 2.20462 * mtom
        self.mass = 0.04674*(self.mtow_lbs**0.397)*(self.S_ft**0.36)*(self.n_ult**0.397)*(self.A**1.712)*0.453592

class FuselageWeight(Component):
    def __init__(self,identifier, mtom, lf, nult, wf, hf, Vc):
        """ Returns fuselage weight, cessna method page 75 Pt 5. component weight estimaation Roskam.

        :param mtom: Maximum take off weight
        :type mtom: float
        :param max_per:  Maximium perimeter of the fuselage
        :type max_per: float
        :param lf: Fuselage length
        :type lf: float
        :param npax: Amount of passengers including pilot
        :type npax: int
        """        
        super().__init__()
        self.id = "fuselage"
        self.mtow_lbs = 2.20462 * mtom
        self.lf_ft, self.lf = lf*3.28084, lf

        self.nult = nult # ultimate load factor
        self.wf_ft = wf*3.28084 # width fuselage [ft]
        self.hf_ft = hf*3.28084 # height fuselage [ft]
        self.Vc_kts = Vc*1.94384449 # design cruise speed [kts]

        self.fweigh_USAF = 200*((self.mtow_lbs*self.nult/10**5)**0.286*(self.lf_ft/10)**0.857*((self.wf_ft+self.hf_ft)/10)*(self.Vc_kts/100)**0.338)**1.1
        self.mass = self.fweigh_USAF*0.453592

        #if identifier == "J1":
        #    self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.max_per_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
        #    self.mass = self.fweight_high*0.453592
        #else:
        #    self.fweight_high = 14.86*(self.mtow_lbs**0.144)*((self.lf_ft/self.max_per_ft)**0.778)*(self.lf_ft**0.383)*(self.npax**0.455)
        #    self.fweight_low = 0.04682*(self.mtow_lbs**0.692)*(self.max_per_ft**0.374)*(self.lf_ft**0.590)
        #    self.fweight = (self.fweight_high + self.fweight_low)/2
        #    self.mass = self.fweight*0.453592

class LandingGear(Component):
    def __init__(self, mtom):
        """Computes the mass of the landing gear, simplified Cessna method for retractable landing gears pg. 81 Pt V component weight estimation

        :param mtom: maximum take off weight
        :type mtom: float
        """        
        super().__init__()
        self.id = "landing gear"
        self.mtow_lbs = 2.20462 * mtom
        self.mass = (0.04*self.mtow_lbs + 6.2)*0.453592


class Powertrain(Component):
    def __init__(self,p_max, p_dense ):
        """Returns the mas of the engines based on power

        :param p_max: Maximum power [w]
        :type p_max: float
        :param p_dense: Power density [w/kg]
        :type p_dense: float
        """        
        super().__init__()
        self.id = "Powertrain"
        self.mass = 12 * (13 + 10) #p_max/p_dense 12 engines (13 kg) and inverters (10 kg) These are scimo engines and converters https://sci-mo.de/motors/


class Propeller(Component):
    def __init__(self ):
        """Returns the mas of the engines based on power

        :param p_max: Maximum power [w]
        :
        """
        
        super().__init__()
        self.id = "Propeller"
        self.mass = 6 * 20 # 6 propellers and 30 kg per proppeller (I just googled a bit)

class HorizontalTailWeight(Component):
    def __init__(self, w_to, S_h, A_h, t_r_h ):
        """Computes the mass of the horizontal tail, only used for Joby. Cessna method pg. 71 pt V component weight estimation

        :param W_to: take off weight in  kg
        :type W_to: float
        :param S_h: Horizontal tail area in  m^2
        :type S_h: float
        :param A_h: Aspect ratio horizontal tail
        :type A_h: float
        :param t_r_h: Horizontal tail maximum root thickness in m 
        :type t_r_h: float
        """        

        self.id = "Horizontal tail"
        w_to_lbs = 2.20462262*w_to
        S_h_ft = 10.7639104*S_h
        t_r_h_ft = 3.2808399*t_r_h

        super().__init__()
        self.mass =  (3.184*w_to_lbs**0.887*S_h_ft**0.101*A_h**0.138)/(174.04*t_r_h_ft**0.223)*0.45359237

class NacelleWeight(Component):
    def __init__(self, p_to):
        """ Returns nacelle weight

        :param w_to: Total take off weight aka MTOM
        :type w_to: float
        """        
        super().__init__()
        self.id = "Nacelles"
        self.p_to_hp = 0.001341*p_to
        self.mass = 0.24*self.p_to_hp*0.45359237 # Original was 0.24 but decreased it since the electric aircraft would require less structural weight0

class H2System(Component):
    def __init__(self, energy, cruisePower, hoverPower):
        """Returns the lightest solutions of the hydrogen 

        :param energy: Amount of energy consumed
        :type energy: float
        :param cruisePower: Power during cruise
        :type cruisePower: float
        :param hoverPower: Power during hover
        :type hoverPower: float
        """        
        raise Exception("This function is deprecated and no longer suppored, only here for the sake of documentation")
        super().__init__()
        self.id = "Hydrogen system"
        echo = np.arange(0,1.5,0.05)

        #batteries
        Liionbat = BatterySizing(sp_en_den= 0.3, vol_en_den=0.45, sp_pow_den=2,cost =30.3, charging_efficiency= const.ChargingEfficiency, depth_of_discharge= const.DOD, discharge_effiency=0.95)
        Lisulbat = BatterySizing(sp_en_den= 0.42, vol_en_den=0.4, sp_pow_den=10,cost =61.1, charging_efficiency= const.ChargingEfficiency, depth_of_discharge= const.DOD, discharge_effiency=0.95)
        Solidstatebat = BatterySizing(sp_en_den= 0.5, vol_en_den=1, sp_pow_den=10,cost =82.2, charging_efficiency= const.ChargingEfficiency, depth_of_discharge= const.DOD, discharge_effiency=0.95)
        #HydrogenBat = BatterySizing(sp_en_den=1.85,vol_en_den=3.25,sp_pow_den=2.9,cost=0,discharge_effiency=0.6,charging_efficiency=1,depth_of_discharge=1)


        #-----------------------Model-----------------
        BatteryUsed = Liionbat
        FirstFC = FuellCellSizing(const.PowerDensityFuellCell,const.VolumeDensityFuellCell,const.effiencyFuellCell, 0)
        FuelTank = HydrogenTankSizing(const.EnergyDensityTank,const.VolumeDensityTank,0)
        InitialMission = MissionRequirements(EnergyRequired= energy, CruisePower= cruisePower, HoverPower= hoverPower )


        #calculating mass
        Mass = PropulsionSystem.mass(np.copy(echo),
                                                                    Mission= InitialMission, 
                                                                    Battery = BatteryUsed, 
                                                                    FuellCell = FirstFC, 
                                                                    FuellTank= FuelTank)

        TotalMass, TankMass, FuelCellMass, BatteryMas, coolingmass= Mass

        # OnlyH2Tank, OnlyH2FC, OnlyH2TankVol, OnlyH2FCVol = onlyFuelCellSizing(InitialMission, FuelTank, FirstFC)

        self.mass = np.min(TotalMass)


class Miscallenous(Component):
    def __init__(self, mtom, oew, npax) -> None:
        """ Returns the miscallenous weight which consists out of flight control, electrical system
        , avionics, aircondition and furnishing. All in line comments refer to pages in
        Pt. 5 Component weight estimation by Roskam

        :param mtom: Maximum take-off weight
        :type mtom: float
        """        
        super().__init__()
        self.id = "misc"
        w_to_lbs = 2.20462262*mtom
        w_oew_lbs = 2.20462262*oew

        mass_fc = 0.0168*w_to_lbs # flight control system weight Cessna method pg. 98
        mass_elec = 0.0268*w_to_lbs # Electrical system mass  cessna method pg. 101
        mass_avionics = 40 + 0.008*w_to_lbs # Avionics system mass Torenbeek pg. 103
        mass_airco = 0.018*w_oew_lbs   # Airconditioning mass Torenbeek method pg. 104
        mass_fur = 0.412*npax**1.145*w_to_lbs**0.489 # Furnishing mass Cessna method pg.107

        self.mass = np.sum([mass_fur, mass_airco, mass_avionics, mass_elec, mass_fc])*0.45359237


        
def get_weight_vtol(perf_par: AircraftParameters, fuselage: Fuselage, wing: Wing,  engine: Engine, vtail: VeeTail, test= False):
    """ This function is used for the final design, it reuses some of the codes created during
    the midterm. It computes the final weight of the vtol using the data structures created in the
    final design phase

    It uses the following weight components
    --------------------------------------
    Powersystem mass -> Sized in power sizing, retrieved from perf class
    Engine mass -> Scimo engines and inverters used
    wing mass -> class II/wingbox code
    vtail mass -> Class II/wingbox code
    fuselage mass -> Class II
    landing gear mass -> Class II
    nacelle mass -> class II
    misc mass -> class II
    --------------------------------------
    """    


    # Wing mass 
    wing.wing_weight = WingWeight(perf_par.MTOM, wing.surface, perf_par.n_ult, wing.aspect_ratio).mass #This is automatically updated in the wing box calculations if they work

    # Vtail mass
    # Wing equation is used instead of horizontal tail because of the heay load of the engine which is attached
    vtail.vtail_weight = WingWeight(perf_par.MTOM, vtail.surface, perf_par.n_ult, vtail.aspect_ratio).mass

    #fuselage mass
    fuselage.fuselage_weight = FuselageWeight("J1", perf_par.MTOM, fuselage.length_fuselage, perf_par.n_ult, fuselage.width_fuselage_outer, fuselage.height_fuselage_outer, const.v_cr).mass

    #landing gear mass
    perf_par.lg_mass = LandingGear(perf_par.MTOM).mass

    # Nacelle and engine mass

    total_engine_mass = Powertrain(perf_par.hoverPower, const.p_density).mass + Propeller().mass + 90 #90 kg is for the pylon length
    nacelle_mass = NacelleWeight(perf_par.hoverPower).mass

    engine.totalmass = nacelle_mass + total_engine_mass
    engine.mass_perpowertrain = (engine.totalmass)/const.n_engines
    engine.mass_pernacelle = nacelle_mass/const.n_engines
    engine.mass_pertotalengine = total_engine_mass/const.n_engines

    # Misc mass
    perf_par.misc_mass = Miscallenous(perf_par.MTOM, perf_par.OEM, const.npax + 1).mass

    perf_par.OEM = np.sum([ perf_par.powersystem_mass ,   wing.wing_weight, vtail.vtail_weight, fuselage.fuselage_weight, nacelle_mass, total_engine_mass, perf_par.lg_mass, perf_par.misc_mass])
    perf_par.MTOM =  perf_par.OEM + const.m_pl

    # Update weight not part of a data structure

    return perf_par, wing, vtail, fuselage, engine


