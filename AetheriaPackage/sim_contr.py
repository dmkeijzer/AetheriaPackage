import numpy as np
from scipy.interpolate import RegularGridInterpolator,interp1d
import numpy as np
import scipy.linalg as lg
import itertools
from warnings import warn
from AetheriaPackage.data_structs import *
import AetheriaPackage.GeneralConstants as const
import matplotlib.pyplot as plt
import control.matlab as ml
import control as cl

def loading_diagram(wing_loc, lf, fuselage, wing, vtail, aircraft, power, engine):
    """_summary_

    :param wing_loc: _description_
    :type wing_loc: _type_
    :param lf: _description_
    :type lf: _type_
    :param fuselage: _description_
    :type fuselage: _type_
    :param wing: _description_
    :type wing: _type_
    :param vtail: _description_
    :type vtail: _type_
    :param aircraft: _description_
    :type aircraft: _type_
    :param power: _description_
    :type power: _type_
    :param engine: _description_
    :type engine: _type_
    :return: _description_
    :rtype: _type_
    """    

    dict_mass_loc = { 
        "wing": (wing.wing_weight, wing_loc),
        "vtail": (vtail.vtail_weight, wing_loc + vtail.length_wing2vtail),
        "engine": (engine.totalmass, (4*(wing_loc - 2) + 2*(wing_loc + vtail.length_wing2vtail))/(6)),
        "fuel_cell": (FuelCell.mass, fuselage.length_cockpit + fuselage.length_cabin + FuelCell.depth/2),
        "battery": (power.battery_mass, power.battery_pos), # Battery was placed in the wing
        "cooling": (power.cooling_mass, fuselage.length_cockpit +  fuselage.length_cabin + fuselage.length_tank), # Battery was placed in the wing
        "tank": (power.h2_tank_mass, fuselage.length_cockpit +  fuselage.length_cabin + fuselage.length_tank/2 ), # Battery was placed in the wing
        "landing_gear": (aircraft.lg_mass,  lf*const.cg_fuselage ), # For now, assume it coincides with the cg of the fuselage
        "fuselage": (fuselage.fuselage_weight, lf*const.cg_fuselage),
        "misc": (aircraft.misc_mass, lf*const.cg_fuselage),
    }


    oem_mass = np.sum(x[0] for x in dict_mass_loc.values())
    oem_cg = np.sum([x[0]*x[1] for x in dict_mass_loc.values()])/oem_mass

    # Initalize lists anc create set up
    loading_array = np.array([[5.056, 125],  # payload
                    [1.723, 77], # pilot
                    [3.453, 77], # passengers row 1
                    [3.453, 77], # passengers row 1
                    [4.476, 77], # passengers row 2
                    [4.476, 77]]) # passengers row 2

    #------------------ front to back -----------------------------------------
    mass_array = [oem_mass]
    mass_pos_array = [oem_cg]

    for i in loading_array:
        mass_array.append(mass_array[-1] + i[1])
        mass_pos_array.append((mass_array[-1]*mass_pos_array[-1] + i[0]*i[1])/(mass_array[-1] + i[0]))

    #----------------------- back to front -----------------------------------
    mass_array2 = [oem_mass]
    mass_pos_array2 = [oem_cg]

    for j in reversed(loading_array[[1,2,3,4,5,0]]):
        mass_array2.append(mass_array2[-1] + j[1])
        mass_pos_array2.append((mass_array2[-1]*mass_pos_array2[-1] + j[0]*j[1])/(mass_array2[-1] + j[0]))

    #------------------------------------ log results --------------------------------------------
    res = {
        "frontcg": min(mass_pos_array), 
        "rearcg": max(mass_pos_array2),
        "oem_cg": oem_cg
        }
        
    res_margin = {
        "frontcg": min(mass_pos_array)-0.1*(max(mass_pos_array2)-min(mass_pos_array)),
        "rearcg": max(mass_pos_array2)+0.1*(max(mass_pos_array2)-min(mass_pos_array)),
        "oem_cg": oem_cg
                 }

    return res, res_margin

def size_vtail_opt(WingClass, FuseClass, VTailClass, StabClass, Aeroclass, AircraftClass, PowerClass, EngineClass, b_ref, stepsize=1e-2,  CLh_initguess = -0.2, CLh_step = 0.02, plot = False):
    """_summary_

    :param WingClass: _description_
    :type WingClass: _type_
    :param FuseClass: _description_
    :type FuseClass: _type_
    :param VTailClass: _description_
    :type VTailClass: _type_
    :param StabClass: _description_
    :type StabClass: _type_
    :param Aeroclass: _description_
    :type Aeroclass: _type_
    :param AircraftClass: _description_
    :type AircraftClass: _type_
    :param PowerClass: _description_
    :type PowerClass: _type_
    :param EngineClass: _description_
    :type EngineClass: _type_
    :param b_ref: _description_
    :type b_ref: _type_
    :param stepsize: _description_, defaults to 1e-2
    :type stepsize: _type_, optional
    :param CLh_initguess: _description_, defaults to -0.6
    :type CLh_initguess: float, optional
    :param CLh_step: _description_, defaults to 0.02
    :type CLh_step: float, optional
    :param plot: _description_, defaults to False
    :type plot: bool, optional
    :return: _description_
    :rtype: _type_
    """    
    CLh = CLh_initguess

    dict_log = {
        "clh_lst": [],
        "S_vee_lst": [],
        "span_vee_lst": [],
        "trim_drag_lst": [],
        "aspect_ratio_lst": [],
        "wing_pos_lst": [],
        "shs_lst": [],
        "ctrl_surf_lst": [],
        "cl_vee_cr_lst": []
    }


    for A_h in np.arange(5, 9, 0.5):
        VTailClass.aspect_ratio = A_h
        CLh = CLh_initguess
        counter = 0
        while True:
            shs, wing_loc, cg_front_bar, cg_aft_bar, cg_dict_margin = wing_location_horizontalstab_size(WingClass, FuseClass,Aeroclass, VTailClass, AircraftClass, PowerClass, EngineClass, StabClass,  A_h, CLh_approach=CLh, stepsize= stepsize)
            l_v = FuseClass.length_fuselage * (1 - wing_loc)
            Vh_V2 = 0.95*(1 + const.axial_induction_factor)**2
            control_surface_data = get_control_surface_to_tail_chord_ratio(WingClass, FuseClass,VTailClass, Aeroclass, CLh, l_v, Cn_beta_req=0.0571, beta_h=1, eta_h=0.95, total_deflection=20 * np.pi / 180, design_cross_wind_speed=5.14, step=0.1 * np.pi / 180)
            CLvee_cr_N = (Aeroclass.cm_ac + Aeroclass.cL_cruise * (abs(cg_aft_bar - cg_front_bar))) / (Vh_V2 * control_surface_data["S_vee"]/WingClass.surface *np.cos(control_surface_data["dihedral"]) * l_v / WingClass.chord_mac)
    
            if type(control_surface_data["control_surface_ratio"]) is str:
                break

            dict_log["clh_lst"].append(CLh)
            dict_log["S_vee_lst"].append(control_surface_data["S_vee"])
            dict_log["span_vee_lst"].append(np.sqrt(A_h * control_surface_data["S_vee"]))
            dict_log["trim_drag_lst"].append(CLvee_cr_N ** 2 * control_surface_data["S_vee"]/A_h)
            dict_log["aspect_ratio_lst"].append(A_h)
            dict_log["wing_pos_lst"].append((shs, wing_loc, cg_front_bar, cg_aft_bar, cg_dict_margin))
            dict_log["shs_lst"].append(shs)
            dict_log["ctrl_surf_lst"].append(control_surface_data)
            dict_log["cl_vee_cr_lst"].append(CLvee_cr_N)

            #Move to next step
            CLh = CLh - CLh_step

            if counter > 10:
                break
            counter += 1

    if plot:
        # Create two subplots side by side
        fig = plt.figure(figsize=(12, 5))

        # First subplot on the left
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(dict_log["clh_lst"], dict_log["aspect_ratio_lst"],dict_log["S_vee_lst"], color='b')
        ax1.set_title('Tail size')
        ax1.set_xlabel("Clh")
        ax1.set_ylabel("Aspect ratio")
        ax1.set_zlabel("S_vee")

        # Second subplot on the right
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(dict_log["clh_lst"], dict_log["aspect_ratio_lst"],dict_log["trim_drag_lst"], color='r')
        ax2.set_title('Trim drag')
        ax2.set_xlabel("Clh")
        ax2.set_ylabel("Aspect ratio")
        ax2.set_zlabel("Trim drag")

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Show the plots
        plt.show()
        
    filter = (dict_log["span_vee_lst"] > b_ref) * (dict_log["shs_lst"] < 1.02*np.min(dict_log["shs_lst"]))
    design_idx = np.argmin(np.array(dict_log["trim_drag_lst"])[filter])

    CLh = dict_log["clh_lst"][design_idx]
    b_vee = dict_log["span_vee_lst"][design_idx]
    Ah = dict_log["aspect_ratio_lst"][design_idx]
    Shs, wing_loc, cg_front_bar, cg_aft_bar, cg_dict= dict_log["wing_pos_lst"][design_idx]
    shs_min = np.min(dict_log["shs_lst"])
    ctrl_surf_data = dict_log["ctrl_surf_lst"][design_idx]
    cl_vee_cr = dict_log["cl_vee_cr_lst"][design_idx]

    if plot:
        print(f"|{CLh=:^55}|")
        print(f"|{b_vee=:^55}|")
        print(f"|{Ah=:^55}|")
        print(f"|{Shs=:^55}|")
        print(f"|{shs_min=:^55}|")
        print(f"|{wing_loc=:^55}|")

    l_v = FuseClass.length_fuselage * (1 - wing_loc)
    Vh_V2 = 0.95*(1+const.axial_induction_factor)**2

    AircraftClass.oem_cg = cg_dict["oem_cg"]
    AircraftClass.wing_loc = wing_loc
    AircraftClass.cg_front = cg_dict["frontcg"]
    AircraftClass.cg_rear = cg_dict["rearcg"]
    AircraftClass.cg_front_bar = cg_front_bar
    AircraftClass.cg_rear_bar =  cg_aft_bar
    StabClass.cg_front_bar = cg_front_bar
    StabClass.cg_rear_bar =  cg_aft_bar
    
    VTailClass.cL_cruise = cl_vee_cr 
    VTailClass.max_clh = CLh 
    VTailClass.length_wing2vtail = l_v
    VTailClass.rudder_max = np.radians(ctrl_surf_data["max_rudder_angle"])
    VTailClass.elevator_min = np.radians(ctrl_surf_data["min_elevator_angle"])
    VTailClass.dihedral = ctrl_surf_data["dihedral"]
    VTailClass.surface = Shs*WingClass.surface
    VTailClass.shs= Shs
    VTailClass.c_control_surface_to_c_vee_ratio = ctrl_surf_data["control_surface_ratio"]
    VTailClass.ruddervator_efficiency = ctrl_surf_data["tau"]
    VTailClass.span = b_vee
    VTailClass.aspect_ratio = Ah

    StabClass.Cm_de = control_surface_data["cm_de"]
    StabClass.Cn_dr = control_surface_data["cn_dr"]

    return WingClass, FuseClass, VTailClass, StabClass



def stabcg(ShS, x_ac, CLah, CLaAh, depsda, lh, c, VhV2, SM):
    """" 
    Short desription of the gfunction
    """
    x_cg = x_ac + (CLah/CLaAh)*(1-depsda)*ShS*(lh/c)*VhV2 - SM
    return x_cg

def ctrlcg(ShS, x_ac, Cmac, CLAh, CLh, lh, c, VhV2):
    """" 
    Short desription of the gfunction
    """
    x_cg = x_ac - Cmac/CLAh + CLh*lh*ShS * VhV2 / (c * CLAh)
    return x_cg

def CLaAhcalc(CLaw, b_f, b, S, c_root):
    """" 
    Short desription of the gfunction
    """
    CLaAh = CLaw * (1 + 2.15 * b_f / b) * (S - b_f * c_root) / S + np.pi * b_f ** 2 / (2 * S)
    return CLaAh

def x_ac_fus_1calc(b_f, h_f, l_fn, CLaAh, S, MAC):
    """" 
    Short desription of the gfunction
    """
    x_ac_stab_fus1_bar = -(1.8 * b_f * h_f * l_fn) / (CLaAh * S * MAC)
    return x_ac_stab_fus1_bar

def x_ac_fus_2calc(b_f, S, b, Lambdac4, taper, MAC):
    """" 
    Short desription of the gfunction
    """
    x_ac_stab_fus2_bar = (0.273 * b_f * S * (b - b_f) * np.tan(Lambdac4)) / ((1 + taper) * MAC ** 2 * b*(b + 2.15 * b_f))
    return x_ac_stab_fus2_bar

def betacalc(M):
    """" 
    Short desription of the gfunction
    """
    return np.sqrt(1-M**2)

def CLahcalc(A_h, beta, eta, Lambdah2):
    """" 
    Short desription of the gfunction
    """
    CLah = 2 * np.pi * A_h / (2 + np.sqrt(4 + (A_h * beta / eta) ** 2 * (1 + (np.tan(Lambdah2)) ** 2 / beta ** 2)))
    return CLah

def stab_formula_coefs(CLah, CLaAh, depsda, l_h, MAC,Vh_V_2, x_ac_stab_bar, SM):
    """" 
    Short desription of the gfunction
    """
    m_stab = 1 / ((CLah / CLaAh) * (1 - depsda) * (l_h / MAC) * Vh_V_2)
    q_stab = (x_ac_stab_bar - SM) / ((CLah / CLaAh) * (1 - depsda) * (l_h / MAC) * Vh_V_2)
    return m_stab, q_stab

def CLh_approach_estimate(A_h):
    """" 
    Short desription of the gfunction
    """
    CLh_approach = -0.35 * A_h ** (1 / 3)
    return CLh_approach

def cmac_fuselage_contr(b_f, l_f, h_f, CL0_approach, S, MAC, CLaAh):
    """" 
    Short desription of the gfunction
    """
    Cm_ac_fuselage = -1.8 * (1 - 2.5 * b_f / l_f) * np.pi * b_f * h_f * l_f * CL0_approach / (4 * S * MAC * CLaAh)
    return Cm_ac_fuselage

def ctrl_formula_coefs(CLh_approach, CLAh_approach, l_h, MAC, Vh_V_2, Cm_ac, x_ac_stab_bar):
    """" 
    Short desription of the gfunction
    """
    m_ctrl = 1 / ((CLh_approach / CLAh_approach) * (l_h / MAC) * Vh_V_2)
    q_ctrl = ((Cm_ac / CLAh_approach) - x_ac_stab_bar) / ((CLh_approach / CLAh_approach) * (l_h / MAC) * Vh_V_2)
    return m_ctrl, q_ctrl

def downwash_k(lh, b):
    """
    lh: distance from wing ac to horizontal tail [m]
    b: wingspan [m]

    returns
    k: factor in downwash formula
    """
    k = 1 + (1 / (np.sqrt(1 + (lh / b) ** 2))) * (1 / (np.pi * lh / b) + 1)
    return k


def downwash(k, CLa, A):
    """
    k: factor calculated with downwash_k()
    CLa: wing CL-alpha [rad^-1]
    A: wing aspect ratio [-]
    """
    depsda = k * CLa / (np.pi*A)
    return depsda


def wing_location_horizontalstab_size(WingClass, FuseClass, Aeroclass, VtailClass, AircraftClass, PowerClass, EngineClass, StabClass, A_h,  plot=False, CLh_approach = None, stepsize = 0.002, cg_shift = 0):
    """" 
    Short desription of the gfunction
    """
    CLAh_approach = Aeroclass.cL_max * 0.9 # Assumes fuselage contribution negligible

    if CLh_approach == None:
        CLh_approach = CLh_approach_estimate(A_h)

    # Initalise wing placement optimisaton
    dict_log = {
        "wing_loc_lst": [],
        "cg_front_lst": [],
        "cg_rear_lst": [],
        "Shs_min_lst": [],
        "cg_dict_marg_lst": [],
        "stab_lst": [],
        "ctrl_lst": [],
    }

    for wing_loc in np.linspace(0.3, 0.65, np.size(np.arange(-1,2,stepsize))):
        l_h = FuseClass.length_fuselage * (1-wing_loc)
        l_fn = wing_loc * FuseClass.length_fuselage - const.x_ac_stab_wing_bar * WingClass.chord_mac - WingClass.x_lemac
        depsda = downwash(downwash_k(l_fn, WingClass.span), Aeroclass.cL_alpha, WingClass.aspect_ratio) 
        cg_dict, cg_dict_margin = loading_diagram(wing_loc * FuseClass.length_fuselage, FuseClass.length_fuselage, FuseClass, WingClass, VtailClass, AircraftClass, PowerClass, EngineClass )
        cg_front_bar = (cg_dict_margin["frontcg"] - wing_loc * FuseClass.length_fuselage + const.x_ac_stab_wing_bar * WingClass.chord_mac)/ WingClass.chord_mac
        cg_rear_bar = (cg_dict_margin["rearcg"] - wing_loc * FuseClass.length_fuselage + const.x_ac_stab_wing_bar * WingClass.chord_mac)/ WingClass.chord_mac
        CLaAh = CLaAhcalc(Aeroclass.cL_alpha, FuseClass.width_fuselage_outer, WingClass.span, WingClass.surface, WingClass.chord_root)

        # Computing aerodynamic centre
        x_ac_stab_fus1_bar = x_ac_fus_1calc(FuseClass.width_fuselage_outer, FuseClass.height_fuselage_outer, l_fn, CLaAh, WingClass.surface, WingClass.chord_mac)
        x_ac_stab_fus2_bar = x_ac_fus_2calc(FuseClass.width_fuselage_outer, WingClass.surface, WingClass.span, WingClass.quarterchord_sweep, WingClass.taper, WingClass.chord_mac)
        x_ac_stab_bar = const.x_ac_stab_wing_bar + x_ac_stab_fus1_bar + x_ac_stab_fus2_bar + const.x_ac_stab_nacelles_bar

        # Computing moment about the aerodynamic centre
        Cm_ac_fuselage = cmac_fuselage_contr(FuseClass.width_fuselage_outer, FuseClass.length_fuselage, FuseClass.height_fuselage_outer, Aeroclass.cL_alpha0_approach, WingClass.surface, WingClass.chord_mac, CLaAh)  # CLaAh for ctrl is different than for stab if cruise in compressible flow
        Cm_ac = Aeroclass.cm_ac + const.Cm_ac_flaps + Cm_ac_fuselage + const.Cm_ac_nacelles
        
        # computing misc variables required
        beta = betacalc(const.mach_cruise)
        CLah = CLahcalc(A_h, beta, const.eta_a_f, const.sweep_half_chord_tail)

        # Creating actually scissor plot
        cg_bar  = np.linspace(-1,2,2000)
        m_ctrl, q_ctrl = ctrl_formula_coefs(CLh_approach, CLAh_approach, l_h, WingClass.chord_mac, const.Vh_V_2, Cm_ac, x_ac_stab_bar) # x_ac_bar for ctrl is different than for stab if cruise in compressible flow
        m_stab, q_stab = stab_formula_coefs(CLah, CLaAh, depsda, l_h, WingClass.chord_mac, const.Vh_V_2, x_ac_stab_bar, const.stab_margin)
        ShS_stab = m_stab * cg_bar - q_stab
        ShS_ctrl = m_ctrl * cg_bar + q_ctrl

        # retrieving minimum tail sizing
        idx_ctrl = cg_bar == min(cg_bar, key=lambda x:abs(x - cg_front_bar))
        idx_stab = cg_bar == min(cg_bar, key=lambda x:abs(x - cg_rear_bar))
        ShSmin = max(ShS_ctrl[idx_ctrl], ShS_stab[idx_stab])[0]


        if False:

            plt.plot(cg_bar, ShS_stab, label= "stability")
            plt.plot(cg_bar, ShS_ctrl, label= "Control")
            plt.hlines(ShSmin, cg_front_bar, cg_rear_bar)
            plt.annotate(f"{wing_loc=}", (1,0))
            plt.annotate(f"{cg_front_bar=}", (1,-.1))
            plt.annotate(f"{cg_rear_bar=}", (1,-.17))
            plt.annotate(f"{ShSmin=}", (1,-.24))
            plt.legend()
            plt.grid()
            plt.show()

        # Storing results
        dict_log["wing_loc_lst"].append(wing_loc)
        dict_log["cg_front_lst"].append(cg_front_bar)
        dict_log["cg_rear_lst"].append(cg_rear_bar)
        dict_log["Shs_min_lst"].append(ShSmin)
        dict_log["cg_dict_marg_lst"].append(cg_dict_margin)
        dict_log["stab_lst"].append(ShS_stab)
        dict_log["ctrl_lst"].append(ShS_ctrl)


    # Selecting optimum design
    design_idx = np.argmin(dict_log["Shs_min_lst"])
    design_shs = dict_log["Shs_min_lst"][design_idx]
    design_wing_loc = dict_log["wing_loc_lst"][design_idx]
    design_cg_front_bar = dict_log["cg_front_lst"][design_idx]
    design_cg_rear_bar = dict_log["cg_rear_lst"][design_idx]
    design_cg_dict_margin = dict_log["cg_dict_marg_lst"][design_idx]
    design_stab_lst = dict_log["stab_lst"][design_idx]
    design_ctrl_lst = dict_log["ctrl_lst"][design_idx]

    if plot:
        fig, axs = plt.subplots(2,1)

        axs[0].plot(cg_bar, design_stab_lst, label= "stability")
        axs[0].plot(cg_bar, design_ctrl_lst, label= "Control")
        axs[0].hlines(design_shs, design_cg_front_bar, design_cg_rear_bar)
        axs[0].annotate(f"{design_wing_loc=}", (1,0))
        axs[0].annotate(f"{design_cg_front_bar=}", (1,-.3))
        axs[0].annotate(f"{design_cg_rear_bar=}", (1,-.6))
        axs[0].annotate(f"{design_shs=}", (1,-.9))
        axs[0].legend()
        axs[0].grid()


        axs[1].plot(dict_log["wing_loc_lst"], dict_log["Shs_min_lst"])
        axs[1].set_xlabel("Wing Location")
        axs[1].set_ylabel("Shs")
        axs[1].grid()
        plt.show()

    WingClass.x_lewing = design_wing_loc*FuseClass.length_fuselage - 0.24 * WingClass.chord_mac - WingClass.x_lemac
    VtailClass.virtual_hor_surface = design_shs*WingClass.surface

    return design_shs, design_wing_loc, design_cg_front_bar, design_cg_rear_bar, design_cg_dict_margin

def get_K(taper_h, AR_h):
    """" 
    Short desription of the gfunction
    """
    taper_points = np.array([0.25, 0.5, 1])
    aspect_ratio_points = np.array([3, 10])
    data = np.array([[0.61, 0.64, 0.68], [0.74, 0.77, 0.8]])
    interp_func = RegularGridInterpolator((aspect_ratio_points, taper_points), data)

    if not aspect_ratio_points[0] < AR_h < aspect_ratio_points[-1]:
        if AR_h < aspect_ratio_points[0]:
            warn(f"Aspect ratio {AR_h} was out of range, defaulting to {aspect_ratio_points[0]=}", category=RuntimeWarning)
            AR_h = aspect_ratio_points[0]

        if AR_h > aspect_ratio_points[-1]:
            warn(f"Aspect ratio {AR_h} was out of range, defaulting to {aspect_ratio_points[-1]=}", category= RuntimeWarning)
            AR_h = aspect_ratio_points[-1]

    K = interp_func([AR_h, taper_h])[0]
    return float(K)

def get_c_control_surface_to_c_vee_ratio(tau):
    """" 
    Short desription of the gfunction
    """
    ce_c_ratio=np.array([0,0.15,0.3])
    tau_arr=np.array([0,0.35,0.55])
    interp_function=interp1d(tau_arr,ce_c_ratio)
    ce_c_ratio_of_tail=interp_function(tau)
    return float(ce_c_ratio_of_tail)

    
def get_tail_dihedral_and_area(Lambdah2,S_hor,Fuselage_volume,S,b,l_v,AR_h,taper_h,Cn_beta_req=0.0571,beta_h=1,eta_h=0.95):
    """" 
    Short desription of the gfunction
    """
    Cn_beta_f=-2*Fuselage_volume/(S*b)    
    K=get_K(taper_h,AR_h)
    CL_alpha_N = CLahcalc(AR_h, beta_h, eta_h, Lambdah2)
    S_ver=(Cn_beta_req-Cn_beta_f)/(K*CL_alpha_N)*S*b/l_v   ###Here, the the vertical tail aspect ratio is taken as AR_vee*K = AR_h*K to calculate required vertical tail area
    S_vee=S_ver+S_hor
    v_angle=np.arctan(np.sqrt(S_ver/S_hor))
    return v_angle, S_vee


#YOU ONLY NEED THIS LAST FUNCTION. THE OTHERS ABOVE ARE SUBFUNCTIONS FOR THE NEXT FUNCTION.

def get_control_surface_to_tail_chord_ratio(wing, fuselage, vtail, aero,  CL_h, l_v, Cn_beta_req=0.0571,beta_h=1,eta_h=0.95,total_deflection=20*np.pi/180,design_cross_wind_speed=9,step=0.1*np.pi/180,axial_induction_factor=0.005):
    """" 
    Short desription of the gfunction
    """
    Vh_V2 = 0.95*(1+axial_induction_factor)**2 #assumed

    tau_from_rudder=0     ##Just to initialize loop
    tau_from_elevator=1   ##Just to initialize loop
    elevator_min=-1*np.pi/180
    rudder_max=total_deflection+elevator_min
    v_angle, S_vee= get_tail_dihedral_and_area(const.sweep_half_chord_tail,vtail.virtual_hor_surface ,fuselage.volume_fuselage,wing.surface,wing.span,l_v,vtail.aspect_ratio ,const.taper_hor)
    K = get_K(const.taper_hor,vtail.aspect_ratio)
    CL_alpha_N = CLahcalc(vtail.aspect_ratio, beta_h, eta_h, const.sweep_half_chord_tail)
    while (tau_from_elevator>tau_from_rudder and rudder_max>1*np.pi/180):
                
        Cn_dr_req=-Cn_beta_req*np.arctan(design_cross_wind_speed/const.v_stall)/(rudder_max)
        CL_tail_de_req=(CL_h-CL_alpha_N*(aero.alpha_approach - aero.downwash_angle_stall))/elevator_min
        Cm_de_req_tail=-CL_tail_de_req*(Vh_V2)*(vtail.virtual_hor_surface*l_v/(wing.surface*wing.chord_mac)) ####Get this from CL_de_required, I made this formula --> S_hor or S_vee should be used here
                
        tau_from_rudder=-Cn_dr_req/(K*CL_alpha_N*np.sin(v_angle)*S_vee/wing.surface*l_v/wing.span*Vh_V2)
        tau_from_elevator=-Cm_de_req_tail/(CL_alpha_N*np.cos(v_angle)*S_vee/wing.surface*l_v/wing.chord_mac*Vh_V2)

        elevator_min=elevator_min-step
        rudder_max=rudder_max-step        

    tau=max([tau_from_rudder,tau_from_elevator])
    ###Recalculate Cm_de or Cn_dr as one of them will now be bigger due to choosing tau as the maximum of the two.
    Cm_de=-Vh_V2*tau*l_v/wing.chord_mac*CL_alpha_N*S_vee/wing.surface*np.cos(v_angle)
    Cn_dr=-Vh_V2*tau*l_v/wing.span*K*CL_alpha_N*S_vee/wing.surface*np.sin(v_angle)
        
    if tau>0.55:
        tau = 0.54999999
        warn("Not possible. Lower CLh than in the horizontal tail sizing program")
        c_control_surface_to_c_vee_ratio=get_c_control_surface_to_c_vee_ratio(tau)
    else:
        c_control_surface_to_c_vee_ratio=get_c_control_surface_to_c_vee_ratio(tau)

    dict_output ={
        "max_rudder_angle":  np.degrees(rudder_max),
        "min_elevator_angle":  np.degrees(elevator_min),
        "tau": tau,
        "cm_de": Cm_de,
        "cn_dr": Cn_dr,
        "dihedral": v_angle,
        "S_vee": S_vee,
        "control_surface_ratio": c_control_surface_to_c_vee_ratio,
    }
    return dict_output


def CZ_adot(CLah,Sh,S,Vh_V2,depsda,lh,c):

    CZ_adot=-CLah*Sh/S*Vh_V2*depsda*lh/c

    return CZ_adot


def Cm_adot(CLah,Sh,S,Vh_V2,depsda,lh,c):

    Cm_adot=-CLah*Sh/S*Vh_V2*depsda*(lh/c)**2

    return Cm_adot




def airfoil_to_wing_CLa(cla, A):
    """
    cla: airfoil cl_alpha [rad^-1]
    A: wing aspect ratio [-]

    returns
    cLa: wing cL_alpha [rad^-1]

    works for horizontal and vertical tails as well
    """
    cLa = cla / (1 + cla / (np.pi * A))
    return cLa


def downwash_k(lh, b):
    """
    lh: distance from wing ac to horizontal tail [m]
    b: wingspan [m]

    returns
    k: factor in downwash formula
    """
    k = 1 + (1 / (np.sqrt(1 + (lh / b) ** 2))) * (1 / (np.pi * lh / b) + 1)
    return k


def downwash(k, CLa, A):
    """
    k: factor calculated with downwash_k()
    CLa: wing CL-alpha [rad^-1]
    A: wing aspect ratio [-]
    """
    depsda = k * CLa / (np.pi*A)
    return depsda


def Cma_fuse(Vfuse, S, c):
    """
    Vfuse: fuselage volume [m^3]
    S: wing surface area [m^2]
    c: mean aerodynamic chord [m]

    returns
    Cma_fuse: Cma component from fuselage [rad^-1]
    """
    Cma_fuse = 2 * Vfuse / (S * c)
    return Cma_fuse


def Cnb_fuse(Vfuse, S, b):
    """
    Vfuse: fuselage volume [-]
    S: wing surface area [m^2]
    b: wingspan [m]

    returns
    Cnb_fuse: Cnb component from fuselage [rad^-1]
    """
    Cnb_fuse = -2 * Vfuse / (S * b)
    return Cnb_fuse


def CDacalc(CL0, CLa, A):
    """
    CL0: wing lift at 0 angle of attack [-]
    CLa: wing CL_alpha [rad^-1]
    A: wing aspect ratio [-]

    returns
    CDa: wing CD_alpha [rad^-1]
    """
    CDa = 2 * CL0 * CLa / (np.pi * A)
    return CDa


def Cxa(CL0, CDa):
    """
    CL0: wing lift at 0 angle of attack [-]
    CDa: wing CD_alpha [rad^-1]

    returns
    Cxa: X-force coefficient derivative wrt alpha [rad^-1]
    """
    Cxa = CL0 - CDa
    return Cxa


def Cxq():
    """
    returns
    Cxq: X-force coefficient derivative wrt q
    ALWAYS 0
    """
    return 0


def Cza(CLa, CD0):
    """
    CLa: wing CL_alpha [rad^-1]
    CD0: CD_0 [-]

    returns
    Cza: Z-force coefficient derivative wrt alpha [rad^-1]
    """
    Cza = -CLa - CD0
    return Cza


def Vhcalc(Sh, lh, S, c):
    """
    Sh: surface area of horizontal stabiliser [m^2]
    lh: distance from wing ac to horizontal tail [m]
    S: surface area of wing [m^2]
    c: mean aerodynamic chord [m]

    returns
    Vh: horizontal tail volume coefficient [-]
    """
    Vh = Sh * lh / (S * c)
    return Vh


def Czq(CLah, Vh):
    """
    CLah: Horizontal stabiliser CL_alpha [rad^-1]
    Vh: horizontal tail volume coefficient [-]

    returns
    Czq: Z-force coefficient derivative wrt q [rad^-1]
    """
    Czq = -2 * CLah * Vh
    return Czq


def Cma(CLa, lcg, c, CLah, Vh, depsda, Cmafuse):
    """
    CLa: wing CL_alpha [rad^-1]
    lcg: distance from wing ac to cg [m]
    c: mean aerodynamic chord [m]
    CLah: horizontal stabiliser CL_alpha [rad^-1]
    Vh: horizontal tail volume coefficient [-]
    depsda: downwash gradient [-]
    Cmafuse: fuselage contribution to Cma [rad^-1]

    returns
    Cma: Aircraft Cm coefficient derivative wrt alpha [rad^-1]
    """
    Cma = CLa * lcg / c - CLah * Vh * (1 - depsda) + Cmafuse
    return Cma


def Cmq(CLah, Vh, lh, c, Cmqfuse):
    """
    CLah: horizontal stabiliser CL_alpha [rad^-1]
    Vh: horizontal tail volume coefficient [-]
    lh: distance from wing ac to horizontal tail [m]
    c: mean aerodynamic chord [m]
    Cmqfuse: fuselage contribution to Cmq [rad^-1]

    returns
    Cmq: Aircraft Cm coefficient derivative wrt q [rad^-1]
    """
    Cmq = -2 * CLah * Vh * lh / c + Cmqfuse
    return Cmq


def Vvcalc(Sv, lv, S, b):
    """
    Sv: surface area of vertical tail [m^2]
    lv: distance from wing ac to vertical tail [m]
    S: surface area of wing [m^2]
    b: wingspan [m]

    returns
    Vv: vertical tail volume coefficient [-]
    """
    Vv = Sv * lv / (S * b)
    return Vv


def Cyb(Cnb, b, lv): #Eq 8-16 FD reader
    """
    Sv: surface area of vertical tail [m^2]
    S: surface area of wing [m^2]
    CLav: vertical tail CL_alpha [rad^-1]

    returns
    Cyb: Y-force coefficient derivative wrt sideslip angle [rad^-1]
    """
    return -Cnb * b / lv


def Cyr(Vv, CLav):
    """
    Vv: vertical tail volume coefficient [-]
    CLav: vertical tail CL_alpha [rad^-1]

    returns
    Cyr: Y-force coefficient derivative wrt yaw rate [rad^-1]
    """
    Cyr = 2 * Vv * CLav
    return Cyr


def Cyp():
    """
    returns
    Cyp: Y-force coefficient derivative wrt roll rate
    ALWAYS 0
    """
    return 0


def Clb(CLa, dihedral, taper):
    """
    CLa: wing CL_alpha [rad^-1]
    dihedral: wing dihedral [rad]
    taper: wing taper ratio [-]

    returns
    Clb: roll-moment coefficient derivative wrt sideslip angle [rad^-1]
    """
    Clb = -CLa * dihedral * (1 + 2 * taper) / (6 * (1 + taper))
    return Clb


def Clp(CLa, taper):
    """
    CLa: wing CL_alpha [rad^-1]
    taper: wing taper ratio [-]

    returns
    Clp: roll-moment coefficient derivative wrt roll rate [rad^-1]
    """
    Clp = -CLa * (1 + 3 * taper) / (12 * (1 + taper))
    return Clp


def Clr(CL0):
    """
    CL0: wing CL at 0 angle of attack [-]

    returns
    Clr: roll-moment coefficient derivative wrt yaw rate [rad^-1]
    """
    return CL0 / 4


def Cnp(CL0):
    """
    CL0: wing CL at 0 angle of attack [-]

    returns
    Cnp: yaw-moment coefficient derivative wrt roll rate [rad^-1]
    """
    return -CL0 / 8


def Cnr(CLav, Vv, lv, b):
    """
    CLav: vertical tail CL_alpha [rad^-1]
    Vv: vertical tail volume coefficient [-]
    lv: distance from wing ac to vertical tail [m]
    b: wingspan [m]

    returns
    Cnr: yaw-moment coefficient derivative wrt yaw rate
    """
    Cnr = -2 * CLav * Vv * lv / b
    return (Cnr)

def muc(m,rho,S,c):
        """
        Computes dimensionless mass for symmetric motion mu_c
        :return: mu_c
        """
        return m/(rho*S*c)

def mub(m,rho,S,b):
        """
        Computes dimensionless mass for asymmetric motion mu_b
        :return: mu_b
        """
        return m/(rho*S*b)

def Cz0(W,theta_0,rho,V,S):
    Cz0= -W*np.cos(theta_0)/(0.5*rho*V**2*S)
    return Cz0

def Cx0(W,theta_0,rho,V,S):
    Cx0= W*np.sin(theta_0)/(0.5*rho*V**2*S)
    return Cx0


def longitudinal_derivatives(Aero, Perf, GeneralConst, Wing, VTail, Stab, lcg, theta_0, Cmafuse=None, Cmqfuse=None, CLa=None, CLah=None, depsda=None,
                             CDa=None, Vh=None, Vfuse=None, cla=None, A=None, clah=None,
                             Ah=None, b=None, k=None, Sh=None):
    Aero.load()
    Perf.load()
    Wing.load()
    VTail.load()

    CD = Aero.cd_cruise
    CL = Aero.cL_cruise
    W = Perf.MTOM * GeneralConst.g0
    rho = GeneralConst.rho_cr
    S = Wing.surface
    m = Perf.MTOM
    c = Wing.chord_mac
    lh = VTail.length_wing2vtail
    CL0 = -Aero.cL_alpha * (Aero.alpha_zero_L)
    CD0 = Aero.cd0_cruise
    Vh_V2 = VTail.Vh_V2
    V = GeneralConst.v_cr


    """
    CD: aircraft drag coefficient[-]
    CL: aircraft lift coefficient [-]
    c: mean aerodynamic chord [m]
    lh: distance from wing ac to horizontal tail [m]
    CL0: Wing CL at angle of attack 0 [-]
    CD0: CD_0 [-]
    lcg: distance from wing ac to cg [m]

    Cmafuse: fuselage contribution to Cma [rad^-1]
    Cmqfuse: fuselage contribution to Cmq [rad^-1]
    CLa: Wing CL_alpha [rad^-1]
    CLah: Horizontal tail CL_alpha [rad^-1]
    depsda: downwash gradient [-]
    CDa: CD derivative wrt angle of attack [rad^-1]
    Vh: Horizontal tail volume coefficient [-]
    Vfuse: Volume of fusealge [m^3]
    S: wing surface area [m^2]
    cla: wing airfoil lift coefficient [rad^-1]
    A: wing aspect ratio [-]
    clah: horizontal tail airfoil lift coefficient [rad^-1]
    Ah: Horizontal tail aspect ratio [-]
    b: wingspan [m]
    k: factor for downwash gradient
    Sh: horizontal tail surface area [m^2]
    Vh_V_ratio: Horizontal tail speed to freestream speed ratio [-]
    W: aircraft weight [N]
    rho= air density [kg/m^3]
    g = gravitational acceleration [m/s^2]
    returns
    dict: dictionary containing longitudinal stability derivatives
    """
    if Cmafuse == None:
        assert Vfuse != None, "Missing input: Vfuse"
        assert S != None, "Missing input: S"
        Cmafuse = Cma_fuse(Vfuse, S, c)
    if Cmqfuse == None:
        Cmqfuse = 0
    if CLa == None:
        assert cla != None, "Missing input: cla"
        assert A != None, "Missing input: A"
        CLa = airfoil_to_wing_CLa(cla, A)
    if CLah == None:
        assert clah != None, "Missing input: clah"
        assert Ah != None, "Missing input: Ah"
        CLah = airfoil_to_wing_CLa(clah, Ah)
    if depsda == None:
        if k == None:
            assert b != None, "Missing input:b"
            downwash_k(lh, b)
        assert A != None, "Missing input: A"
        depsda = downwash(k, CLa, A)
    if CDa == None:
        assert A != None, "Missing input: A"
        CDa = CDacalc(CL0, CLa, A)
    if Vh == None:
        assert Sh != None, "Missing input: Sh"
        assert S != None, "Missing input: S"
        Vh = Vhcalc(Sh, lh, S, c)
    Stab.load()
    Stab.Cxa = Cxa(CL0, CDa)
    Stab.Cxq = Cxq()
    Stab.Cza = Cza(CLa, CD0)
    Stab.Czq = Czq(CLah, Vh)
    Stab.Cma = Cma(CLa, lcg, c, CLah, Vh, depsda, Cmafuse)
    Stab.Cmq = Cmq(CLah, Vh, lh, c, Cmqfuse)
    Stab.Cz_adot = CZ_adot(CLah, Sh, S, Vh_V2, depsda, lh, c)
    Stab.Cm_adot = Cm_adot(CLah, Sh, S, Vh_V2, depsda, lh, c)
    Stab.muc = muc(m, rho, S, c)
    Stab.Cxu = -2 * CD
    Stab.Czu = -2 * CL
    # Stab.Cx0 = -CL
    Stab.Cx0 = Cx0(W, theta_0, rho, V, S)
    Stab.Cz0 = Cz0(W, theta_0, rho, V, S)
    Stab.Cmu = 0  #Because the derivative of CL and Ct with respect to the Mach number is essentially 0.

    Stab.dump()


def lateral_derivatives(Perf, GeneralConst, Wing, VTail, Stab, Cnb, CLav=None, Vv=None, CLa=None, clav=None,
                        Av=None, cla=None, A=None, Cn_beta_dot=None,CY_beta_dot=None): #Cnbfuse=None, Vfuse=None
    Perf.load()
    Wing.load()
    VTail.load()
    Stab.load()
    m = Perf.MTOM
    rho = GeneralConst.rho_cr
    Sv = VTail.surface * np.sin(VTail.dihedral)**2 #Eq.14 NASA paper
    lv = VTail.length_wing2vtail
    S = Wing.surface
    b = Wing.span
    dihedral = 0
    taper = Wing.taper
    CL0 = -Aero.cL_alpha * (Aero.alpha_zero_L)
    """
    Cnb: this is the derivative the yaw moment coefficient with respect to sideslip angle beta- [-]
    theta_0: initial pitch angle [rad]
    Sv: vertical tail surface area [m^2]
    lv: distance from wing ac to vertical tail [m]
    S: wing surface area [m^2]
    b: wingspan [m]
    dihedral: wing dihedral angle [rad]
    taper: wing taper ratio [-]
    CL0: wing CL at 0 angle of attack [-]

    CLav: vertical tail CL_alpha [rad^-1]
    Vv: vertical tail volume coefficient [-]
    CLa: wing CL_alpha [rad^-1]
    Cnbfuse: fuselage contribution to Cnb [rad^-1]
    clav: vertical tail airfoil cl_alpha [rad^-1]
    Av: vertical tail aspect ratio [-]
    cla:wing airfoil cl_alpha [rad^-1]
    A: wing aspect ratio [-]
    Vfuse: fuselage volume [m^3]
    W: aircraft weight [N]
    rho= air density [kg/m^3]
    g = gravitational acceleration [m/s^2]

    returns
    dict: dictionary containing lateral stability derivatives
    """

    if CLav == None:
        assert clav != None, "Missing input: clav"
        assert Av != None, "Missing input: Av"
        CLav = airfoil_to_wing_CLa(clav, Av)
    if Vv == None:
        Vv = Vvcalc(Sv, lv, S, b)
    if CLa == None:
        assert cla != None, "Missing input: cla"
        assert A != None, "Missing input: A"
        CLa = airfoil_to_wing_CLa(cla, A)
    # if Cnbfuse == None:
    #     assert Vfuse != None, "Missing input: Vfuse"
    #     Cnbfuse = Cnb_fuse(Vfuse, S, b)
    if Cn_beta_dot == None:
        Cn_beta_dot=0
    if CY_beta_dot == None:
        CY_beta_dot=0

    Stab.Cyb = Cyb(Cnb, b, lv)
    Stab.Cyp = Cyp()
    Stab.Cyr = Cyr(Vv, CLav)
    Stab.Clb = Clb(CLa, dihedral, taper)
    Stab.Clp = Clp(CLa, taper)
    Stab.Clr = Clr(CL0)
    # Stab.Cnb = Cnb(CLav, Vv, Cnbfuse)
    Stab.Cnp = Cnp(CL0)
    Stab.Cnr = Cnr(CLav, Vv, lv, b)
    Stab.Cy_dr = -Stab.Cn_dr * b / lv #Eq 8-41 FD reader
    Stab.Cy_beta_dot = CY_beta_dot
    Stab.Cn_beta_dot = Cn_beta_dot
    Stab.mub = mub(m, rho, S, b)
    Stab.Cnb = Cnb
    Stab.dump()



def eigval_finder_sym(Stab, Iyy, m, c,V0=45,CXq=0,CXde=0,CZde=0, Cmde=-2.617):      #Iyy = 12081.83972
    """
    Iyy: moment of inertia around Y-axis
    m: MTOM
    c: MAC
    long_stab_dervs: dictionary containing all longitudinal stability derivatives + muc

    returns
    array with eigenvalues NON-DIMENSIONALISED
    """
    Stab.load()
    CX0 = Stab.Cx0
    CXa = Stab.Cxa
    CXu = Stab.Cxu
    CZ0 = Stab.Cz0
    CZa = Stab.Cza
    CZu = Stab.Czu
    CZq = Stab.Czq
    CZadot = Stab.Cz_adot
    Cma = Stab.Cma
    Cmq = Stab.Cmq
    Cmu = Stab.Cmu
    Cmadot = Stab.Cm_adot
    muc = Stab.muc

    KY2 = Iyy / (m*c**2)
    # Aeigval = 4 * muc **2 * KY2 * (CZadot - 2 * muc)
    # Beigval = Cmadot * 2 * muc * (CZq + 2 * muc) - Cmq * 2 * muc * (CZadot - 2 * muc) - 2 * muc * KY2 * (CXu * (CZadot - 2*muc) - 2 * muc * CZa)
    # Ceigval = Cma * 2 * muc * (CZq + 2*muc) - Cmadot * (2 * muc * CX0 + CXu * (CZq + 2*muc)) + Cmq * (CXu * (CZadot - 2*muc) - 2*muc*CZa) + 2 * muc*KY2*(CXa*CZu - CZa * CXu)
    # Deigval = Cmu * (CXa*(CZq + 2*muc) - CZ0 * (CZadot - 2 * muc)) - Cma * (2*muc*CX0 + CXu * (CZq + 2*muc)) + Cmadot * (CX0*CXu - CZ0*CZu) + Cmq*(CXu * CZa - CZu * CXa)
    # Eeigval = -Cmu * (CX0 * CXa + CZ0 * CZa) + Cma * (CX0 * CXu + CZ0 * CZu)
    # print(np.roots(np.array([Aeigval, Beigval, Ceigval, Deigval, Eeigval])))
    
    
        
    # Stab.sym_eigvals = list(np.roots(np.array([Aeigval, Beigval, Ceigval, Deigval, Eeigval])))
    # Stab.dump()
    
    C1 = np.zeros((4, 4), dtype=np.float64)
    C2 = np.zeros((4, 4), dtype=np.float64)
    C3 = np.zeros((4, 1), dtype=np.float64)


    C1[0, 0] = - 2 * c * muc / V0 ** 2
    C1[1, 1] = (CZadot - 2 * muc) * c / V0
    C1[2, 2] = - c / V0
    C1[3, 1] = Cmadot * c / V0
    C1[3, 3] = - 2 * muc * KY2 * c ** 2 / V0 ** 2  # Ky2 is already squared

    C2[0, 0] = CXu / V0
    C2[0, 1] = CXa
    C2[0, 2] = CZ0
    C2[0, 3] = CXq * c / V0  # ** 2  # V should probably not be squared. Check derivation: Correct
    C2[1, 0] = CZu / V0
    C2[1, 1] = CZa
    C2[1, 2] = -CX0            # Should probably be negative: Correct
    C2[1, 3] = (CZq + 2 * muc) * c / V0
    C2[2, 3] = c / V0
    C2[3, 0] = Cmu / V0
    C2[3, 1] = Cma
    C2[3, 3] = Cmq * c / V0
    
    C3[0] = CXde  # / v       # There should not be a divided by v: Correct, don't know how that got in
    C3[1] = CZde
    C3[3] = Cmde
        
    
    As=-np.matmul(np.linalg.inv(C1), C2)
    Bs=-np.matmul(np.linalg.inv(C1), C3)
    
    
    
    Cs=np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
    Ds=np.array([[0], [0], [0], [0]])
    
    
    print('Eigenvalues of the symmetric aircraft motion', np.linalg.eigvals(As))
    
    
    sys=ml.ss(As, Bs, Cs, Ds)
    
    #Htf = ml.tf(sys)
    #Hol_s = ml.tf(Htf.num[0][0], Htf.den[0][0])
    #print(Hol_s)
    #ml.sisotool(Hol_s)
    gain_phugoid_damper=0.25 
    
    ####Determine new system
    
    Bs[3]=Bs[3]*gain_phugoid_damper
    
    sys=ml.ss(As, Bs, Cs, Ds)
    
    
    Htf = ml.tf(sys)
    Hol_s = ml.tf(Htf.num[0][0], Htf.den[0][0])
   
    #Note that the poles of the open loop system after adding gain is still
    #the same as before
    
    feedback=np.array([1,0,0,0])
    sys_damped=ml.feedback(sys,feedback)
    A_matrix=sys_damped.A    
    print('Eigenvalues of the symmetric motion after adding phugoid damper', np.linalg.eigvals(A_matrix))

    tstart=0
    tend=100
    t = np.linspace(tstart,tend,1000)
    u=np.ones(1000)
    
    s=ml.tf([1,0],[1])      
    #t, y = cl.forced_response(sys_damped, t, u,[1,0,0,0])
    H_closed=ml.feedback(Hol_s/(0.05*s+1),1)
    print('Poles', ml.pole(H_closed))
    t, y = cl.forced_response(H_closed, t, u)
    t1, y1 = cl.forced_response(sys_ol, t, u)
    plt.plot(t[21:],y[21:])
    plt.plot(t1[21:],y1[0][21:])
    plt.show()
    
    # plt.plot(t,y[0], label='Aircraft response w. phugoid damper')
    # plt.plot(t1,y1[0], label='Aircraft response w/o phugoid damper')
    # plt.xlabel('time (s)')
    # plt.ylabel('aircraft velocity deviation from equilibrium (m/s)')    
    # plt.legend()    
    # plt.show()
    



def eigval_finder_asymm(Stab, Ixx, Izz, Ixz, m, b, CL, V0=45, CYbdot=0, Cnbdot=0,CYda=0, Cldr=0, Cnda=0,Cndr=-0.10335434353099078,CYdr=0.18090,Clda=-0.0977677051397158):   #Ixx = 10437.12494 Izz = 21722.48912

    """
    Ixx: moment of inertia around X-axis
    Izz: moment of inertia around Z-axis
    Ixz: moment of gyration around X-Z
    m: MTOM
    b: wingspan
    CL: cruise CL
    lat_stab_dervs: dictionary containing all lateral stability derivatives + mub

    returns
    array with eigenvalues NON-DIMENSIONALISED
    """
    Stab.load()
    CYb = Stab.Cyb
    CYp = Stab.Cyp
    CYr = Stab.Cyr
    Clb = Stab.Clb
    Clp = Stab.Clp
    Clr = Stab.Clr
    Cnb = Stab.Cnb
    Cnp = Stab.Cnp
    Cnr = Stab.Cnr
    mub = Stab.mub


    KX2 = Ixx / (m*b**2)
    KZ2 = Izz / (m*b**2)
    KXZ = Ixz / (m*b**2)
    # Aeigval = 16 * mub ** 3 * (KX2 * KZ2 - KXZ ** 2)
    # Beigval = -4 * mub ** 2 * (
    #             2 * CYb * (KX2 * KZ2 - KXZ ** 2) + Cnr * KX2 + Clp * KZ2 + (
    #                 Clr + Cnp) * KXZ)
    # Ceigval = 2 * mub * ((CYb * Cnr - CYr * Cnb) * KX2 + (
    #             CYb * Clp - Clb * CYp) * KZ2 + ((CYb * Cnp - Cnb * CYp) + (
    #             CYb * Clr - Clb * CYr)) * KXZ + 4 * mub * Cnb * KX2 + 4 * mub * Clb * KXZ + 0.5 * (
    #                                  Clp * Cnr - Cnp * Clr))
    # Deigval = -4 * mub * CL * (Clb * KZ2 + Cnb * KXZ) + 2 * mub * (
    #             Clb * Cnp - Cnb * Clp) + 0.5 * CYb * (
    #                       Clr * Cnp - Cnr * Clp) + 0.5 * CYp * (
    #                       Clb * Cnr - Cnb * Clr) + 0.5 * CYr * (
    #                       Clp * Cnb - Cnp * Clb)
    # Eeigval = CL * (Clb * Cnr - Cnb * Clr)
    # print(np.roots(np.array([Aeigval, Beigval, Ceigval, Deigval, Eeigval])))
    
    
    
    Ca=np.array([[1,0,0,0],[0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1]])

    Da=np.array([[0,0],[0,0],
                           [0,0],
                           [0,0]])


    C1a=np.array([[(CYbdot-2*mub)*b/V0,0,0,0],        #Checked
                            [0,-0.5*b/V0 ,0,0],        #Checked
                            [0,0,-2*mub*KX2*(b/V0)**2,2*mub*KXZ*(b/V0)**2],   #Checked
                            [Cnbdot*b/V0,0,2*mub*KXZ*(b/V0)**2,-2*mub*KZ2*(b/V0)**2]])  #Checked

    C2a=np.array([[CYb,CL,CYp*b/(2*V0),(CYr-4*mub)*b/(2*V0)],        #Checked
                            [0,0,b/(2*V0),0],                     #Checked
                            [Clb,0,Clp*b/(2*V0),Clr*b/(2*V0)],  #Checked
                            [Cnb,0,Cnp*b/(2*V0),Cnr*b/(2*V0)]])     #Checked

    C3a=np.array([[CYda,CYdr],     #Checked
                  [0,0],               #Checked
                  [Clda, Cldr],    #Checked
                  [Cnda, Cndr]])   #Checked

    Aa=-np.matmul(np.linalg.inv(C1a),C2a)

    Ba=-np.matmul(np.linalg.inv(C1a),C3a)
    print('Asymmetric motion eigenvalues before yaw damper', np.linalg.eigvals(Aa))
    
    ####Determine gain
    Says=ml.ss(Aa, Ba, Ca, Da)
    Hset = ml.tf(Says)
    Hol = ml.tf(Hset.num[3][1], Hset.den[3][1])
    #ml.sisotool(-Hol)
    gain_yaw_damper=-1
    
    ####Determine new system
    Ba[3,1]=Ba[3,1]*gain_yaw_damper
    Ba[2,1]=Ba[2,1]*gain_yaw_damper
    Ba[2,1]=Ba[1,1]*gain_yaw_damper
    Ba[0,1]=Ba[0,1]*gain_yaw_damper
    
    Says=ml.ss(Aa, Ba, Ca, Da)
    
    Htf = ml.tf(Says)
    Hol_s = ml.tf(Htf.num[0][0], Htf.den[0][0])
    gm, pm, _, _ = ml.margin(Hol_s)
    print("Gain Margin: ", gm)
    print("Phase Margin: ", pm)
    #The phase margin is negative due to the instability of the spiral mode.
    
    
    feedback=np.array([[0,0,0,0],[0,0,0,1]])
    
    #feedback=np.array([[0,0,0,0],[0,0,0,gain_yaw_damper]])
    Says_damped=ml.feedback(Says,feedback)
    A_matrix=Says_damped.A
    
    print('Asymmetric motion eigenvalues after yaw damper', np.linalg.eigvals(A_matrix))

    tstart=-1.1
    tend=20
    t = np.linspace(tstart,tend,10000)
    u=np.array([np.zeros(10000),np.zeros(10000)])
    
       
    # t, y = cl.forced_response(Says_damped, t, u, [0,0,0,0.523])
    # t1, y1 = cl.forced_response(Says, t, u, [0,0,0,0.523])
    # plt.plot(t,y[3],label='Aircraft response w. yaw damper')
    # plt.plot(t1,y1[3], label='Aircraft response w/o yaw damper')
    # plt.xlabel('time (s)')
    # plt.ylabel('aircraft sideslip angle deviation from equilibrium (rad)')    
    # plt.legend()    
    # plt.show()

    

    ####See if equivalent:   YES INDEED 
    s=ml.tf([1,0],[1])
    H_closed=ml.feedback(1/(0.05*s+1)*Hol*(-7/s+gain_yaw_damper),1)
    print('This is what happened when we make a PI controller', ml.pole(H_closed))
    #####These poles show the eigenvalues, except the one at e-14 which is a floating point error.
    t, y = cl.forced_response(H_closed, t, np.zeros(10000),-0.01)
    #H_closed=ml.tf([0.5772,4.461,2.947],[0.05,1.04,1.403,4.846,3.177])
    t1, y1 = cl.forced_response(Says, t, u, [0,0,0,0.01])
    # plt.plot(t1[630:],y1[3][630:], label='Aircraft response w/o yaw damper')
    # plt.plot(t[630:],y[630:],label='Aircraft response w. yaw damper' )
    # plt.xlabel('time (s)')
    # plt.ylabel('aircraft yaw rate (rad/s)')  
    # plt.legend()
    # plt.show()

 

def span_vtail(r, w, g):
    """ Computes the span of the vtail based of the propellor radius, width of the fuselage and dihedral of the vtail

    :param r: propellor radius
    :type r: float
    :param w: width of the fuselage
    :type w: float
    :param g: dihedral of vtail
    :type g: float
    :return: span vtail
    :rtype: float
    """    
    l = np.linspace(0, 2, 1000)
    formula = (w/2 - (l + r)*np.cos(g))**2 + ((l+r)*np.sin(g))**2 - r**2
    # Bisection method
    warn("A built-in method could be used which would likely be much faster")

    tolerance = 1e-6
    a = l[0]
    b = l[-1]
    while b - a > tolerance:
        c = (a + b) / 2
        fc = (w/2 - (c + r)*np.cos(g))**2 + ((c+r)*np.sin(g))**2 - r**2
        if fc == 0:
            # Found exact zero-crossing
            return c
        elif np.sign(fc) == np.sign(formula[0]):
            # Zero-crossing lies between a and c
            a = c
        else:
            # Zero-crossing lies between c and b
            b = c
    # Return the approximate zero-crossing
    if np.tan(g)*w/2 < r:
        s = (a+b)/2 + r
    else:
        s = r/np.sin(g)
    return s


def acai(Bf, fcmin, fcmax, Tg):
    """ Computes the Available Control Authority Index, based on MATLAB original code

    :param Bf: Control effectiveness matrix [-]
    :type Bf: numpy array (as many rows as states - usually 4: vertical thrust, pitch, roll, yaw - and as many columns
            as rotors)
    :param fcmin: Minimum available thrust per rotor [N]
    :type fcmin: numpy array (shape = (n_rotors,1))
    :param fcmax: Maximum available thrust per rotor [N]
    :type fcmax: numpy array (shape = (n_rotors,1))
    :param Tg: External forces acting on body [N]
    :type Tg: numpy array (shape = (n_states,1))
    :return: Available Control Authority Index
    :rtype: float
    """
    sz = np.shape(Bf)
    n = sz[0]
    m = sz[1]
    M = np.arange(m)
    S1 = np.array(list(itertools.combinations(M, n-1)))
    sm = np.shape(S1)[0]
    fc = (fcmin + fcmax)/2
    Fc = Bf @ fc
    choose = S1[0,:]
    B_1j = Bf[:,choose]
    z_jk = (fcmax-fcmin)/2
    z_jk = np.delete(z_jk, choose, 0)
    kesai = lg.null_space(B_1j.T)
    kesai = kesai[:,0]
    B_2j = np.copy(Bf)
    B_2j = np.delete(B_2j, choose, 1)
    E = kesai.T @ B_2j
    dmin = np.zeros((sm,1))
    dmax = np.abs(E) @ z_jk
    temp = dmax - np.abs(kesai.T@(Fc - Tg))
    dmin[0,0] = temp
    for j in np.arange(1, sm):
        choose = S1[j,:]
        B_1j = Bf[:,choose]
        z_jk = (fcmax - fcmin) / 2
        z_jk = np.delete(z_jk, choose, 0)
        kesai = lg.null_space(B_1j.T)
        kesai = kesai[:, 0]
        B_2j = np.copy(Bf)
        B_2j = np.delete(B_2j, choose, 1)
        E = kesai.T @ B_2j
        dmax = np.abs(E) @ z_jk
        temp = dmax - np.abs(kesai.T@(Fc - Tg))
        dmin[j,0] = temp
    if np.min(dmin)>=0:
        degree = np.min(dmin)
    else:
        degree = -np.min(np.abs(dmin))
    return degree

def create_rotor_loc(wingspan, prop_radius, Vtailspan, Vtail_dihedral, x_lewing, root_chord, tip_chord, LE_sweep, l_fus, Vtail_chord):
    """ Creates the rotor_loc array (ONLY VALID FOR AETHERIA CONCEPT)

        :param wingspan: Wing span [m]
        :type wingspan: float
        :param prop_radius: Radius of propellers [m]
        :type prop_radius: float
        :param Vtailspan: V-tail-long span (not projected) [m]
        :type Vtailspan: float
        :param Vtail_dihedral: Dihedral angle of V-tail (upwards positive) [rad]
        :type Vtail_dihedral: float
        :param x_lewing: x-location of root chord's leading edge (distance from nose) [m]
        :type x_lewing: float
        :param root_chord: Main wing's root chord length [m]
        :type root_chord: float
        :param tip_chord: Main wing's tip chord length [m]
        :type tip_chord: float
        :param LE_sweep: Measured sweep angle at main wing's leading edge [rad]
        :type LE_sweep: float
        :param l_fus: Fuselage length [m]
        :type l_fus: float
        :param Vtail_chord: Chord length of V-tail [m]
        :type Vtail_chord: float
        :return: rotor-locations array (first row: x-locations, second row: y-locations, third row: rotation direction)
        :rtype: numpy array
        """
    WOy = wingspan / 2 #y-loc Wing-Outer propeller (wingtip)
    WIy = WOy - 2*prop_radius -0.1 #y-loc Wing-Inner propeller (wingtip - propeller diameter - prop-prop clearance)
    Ty = Vtailspan * np.cos(Vtail_dihedral) / 2 #y-loc Tail propeller (tail wingtip)
    WOx = x_lewing + 0.25 * root_chord - 0.05 * tip_chord #x-loc Wing-Outer propeller (20% tip chord)
    WIx = x_lewing + WIy * np.sin(LE_sweep) #x-loc Wing-Inner propeller (LE point at WIy)
    Tx = l_fus + 0.25*Vtail_chord #x-loc Tail propeller (50% tip chord Vtail)
    rotor_loc = np.array([[WIx, WIx, WOx, WOx, Tx, Tx],
                          [WIy, -WIy, WOy, -WOy, Ty, -Ty],
                          [-1,1,-1,1,1,-1]])
    return rotor_loc


def cg_range_calc(convergence_dir, x_cg_try, x_cg_lim, rotor_loc , rotor_eta, rotor_ku, max_T_per_rotor, Tg):
    """ Finds by convergence the most limiting cg location (vertical flight mode) in a certain convergence direction
        using ACAI to determine controllability given a certain cg location

        :param convergence_dir: -1 to converge to front cg limit, 1 to converge to aft cg limit
        :type convergence_dir: int
        :param x_cg_try: Initial guess of limiting CG location. To converge to front CG location, the guess should be
                         placed more aft, to converge to aft CG location, the guess should be placed more forward [m]
        :type x_cg_try: float
        :param x_cg_lim: Limit after which we do not look for limiting cg locations anymore (to be read as horizontal
                         flight mode cg limit) [m]
        :type x_cg_lim: float
        :param rotor_loc: Array containing x and y locations for rotors + rotation directions (as 1 or -1) [m, m, -]
        :type rotor_loc: numpy array of shape (3, n_rotors)
        :param rotor_eta: 1-D array containing efficiency of each rotor (1=nominal operation) [-]
        :type rotor_eta: numpy array of shape (n_rotors)
        :param rotor_ku: 1-D array containing Torque/Thrust ratios for each rotor [m]
        :type rotor_ku: numpy array of shape (n_rotors)
        :param max_T_per_rotor: Maximum thrust that 1 rotor can provide [N]
        :type max_T_per_rotor: float
        :param Tg: External forces acting on body [N]
        :type Tg: numpy array of shape (n_states,1)
        :return: limiting CG location for vertical flight mode in one convergence direction
        :rtype: float
        """
    step_size = 0.01
    alist = []
    cond = False
    broken = False
    while convergence_dir*x_cg_try <= convergence_dir*x_cg_lim:
        alist.append(x_cg_try)
        rotor_d = np.array([])
        rotor_angle = np.array([])
        rotor_loc[0,:] = rotor_loc[0,:] - x_cg_try
        for i in range(np.shape(rotor_loc)[1]):
            rotor_d = np.append(rotor_d, np.linalg.norm(rotor_loc[:2,i]))
            rotor_angle = np.append(rotor_angle, np.arctan2(rotor_loc[1,i], rotor_loc[0,i]))
        bt = np.copy(rotor_eta)
        bl = -rotor_d * np.sin(rotor_angle) * rotor_eta
        bm = rotor_d * np.cos(rotor_angle) * rotor_eta
        bn = rotor_loc[2,:] * rotor_ku * rotor_eta
        Bf = np.array([bt, bl, bm, bn])
        fcmin = np.zeros((np.shape(rotor_loc)[1], 1))
        fcmax = max_T_per_rotor * np.ones((np.shape(rotor_loc)[1], 1))
        delta = 1e-10
        ACAI = acai(Bf, fcmin, fcmax, Tg)
        if -delta<ACAI<delta:
            ACAI = 0
        if ACAI > 0:
            cond = True
        if ACAI <= 0 and cond:
            broken = True
            break
        x_cg_try = x_cg_try + convergence_dir*step_size
    if not cond:
        return convergence_dir * np.inf
    elif broken:
        return alist[-2]
    else:
        return alist[-1]

def pylon_calc(Wing, Veetail, Fuselage, Stability, AircraftParameters, rotor_loc, pylon_step = 0.1, Tfac_step = 0.05):
    """ Returns a matrix with the Tfac that corresponds to a certain pylon length to ensure vertical flight
        controllability, where Tfac is the factor corresponding to the required maximum thrust per rotor, divided
        by the nominal thrust per rotor (m*g/n_rotors)

        :param Wing: Wing class 
        :type Wing: Wing class (must contain attributes: surface, x_lewing, x_lemac, chord_mac)
        :param Veetail: Veetail class
        :type Veetail: Veetail class (must contain attributes: surface, dihedral, span, length_wing2vtail)
        :param Fuselage: Fuselage class
        :type Fuselage: Fuselage class (must contain attributes: length_fuselage, length_tail, bf)
        :param Stability: Stability class
        :type Stability: Stability class (must contain attributes: cg_front_bar, cg_rear_bar)
        :param AircraftParameters: AircraftParameters class
        :type AircraftParameters: AircraftParameters class (must contain attributes: MTOM)
        :param rotor_loc: Array containing x and y locations for rotors + rotation directions (as 1 or -1) [m, m, -]
        :type rotor_loc: numpy array of shape (3, n_rotors)
        :param pylon_step: Step to loop pylon length [m]
        :type pylon_step: float
        :param Tfac_step: Step to loop Tfac [-]
        :type Tfac_step: float
        :return: Array containing corresponding values of Tfac for different pylon lengths
        :rtype: numpy array of shape(undefined, 2)
        """
    
    wing_surface = Wing.surface
    vtail_surface = Veetail.surface
    vtail_dihedral = Veetail.dihedral
    l_fus = Fuselage.length_fuselage
    length_tail = Fuselage.length_tail
    width_fus = Fuselage.bf
    l_v = Veetail.length_wing2vtail
    b_vee = Veetail.span
    cg_front_lim = Wing.x_lewing + Wing.x_lemac + Wing.chord_mac * Stability.cg_front_bar
    cg_rear_lim = Wing.x_lewing + Wing.x_lemac + Wing.chord_mac * Stability.cg_rear_bar
    ma = AircraftParameters.MTOM
    Sproj = wing_surface + vtail_surface * np.cos(vtail_dihedral) + (l_fus - 0.5*length_tail) * width_fus
    g0 = 9.80665
    rho0 = 1.225

    Tg = np.array([[ma * g0 + rho0 * 11.1**2 * Sproj],#Assumes CD = 2.0
                   [0.5 * rho0 * (31/3.6)**2 * vtail_surface * np.sin(vtail_dihedral) * b_vee * np.sin(vtail_dihedral)/2],
                   [0],
                   [0.5 * rho0 * (31/3.6)**2 * vtail_surface * np.sin(vtail_dihedral) * l_v]])
    r_ku = 0.1 * np.ones(np.shape(rotor_loc)[1])
    r_eta = np.ones(np.shape(rotor_loc)[1])
    loopforpylonsize = True
    log = np.zeros((0,2))
    pylonsize = 0

    while loopforpylonsize:
        loopfortfac = True
        Tfac = 2
        print(f"{pylonsize=}")
        while loopfortfac:
            x_cg_fw = np.array([])
            x_cg_r = np.array([])
            loopforbrokenengine = True
            engineidx = 0
            maxT = Tfac * ma * g0 / np.shape(rotor_loc)[1]
            while loopforbrokenengine:
                x_cg_fw= np.append(x_cg_fw, cg_range_calc(-1, cg_rear_lim, cg_front_lim, rotor_loc, r_eta, r_ku, maxT, Tg))
                x_cg_r= np.append(x_cg_r, cg_range_calc(1, cg_front_lim, cg_rear_lim, rotor_loc, r_eta, r_ku, maxT, Tg))
                r_eta = np.ones(np.shape(rotor_loc)[1])
                r_eta[engineidx] = 0.5 #Assumes 2 engines / prop
                if engineidx == np.shape(rotor_loc)[1] - 1:
                    loopforbrokenengine = False
                engineidx +=1
            if -10 < np.max(x_cg_fw) <= cg_front_lim and 100 > np.min(x_cg_r) >= cg_rear_lim:
                loopfortfac = False
            elif Tfac > 6 :
                loopfortfac = False
                Tfac = np.inf
            else:
                Tfac += Tfac_step
        log = np.vstack((log, [Tfac, pylonsize]))
        pylonsize += pylon_step
        rotor_loc[0,:2] = rotor_loc[0,:2] - pylon_step
        
        if pylonsize > 8:
            loopforpylonsize = False
    return log



