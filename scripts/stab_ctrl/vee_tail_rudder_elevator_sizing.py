from scipy.interpolate import RegularGridInterpolator,interp1d
import numpy as np
import sys
import os
import pathlib as pl
sys.path.append(str(list(pl.Path(__file__).parents)[2]))
os.chdir(str(list(pl.Path(__file__).parents)[2]))
from modules.stab_ctrl.wing_loc_horzstab_sizing import CLahcalc



#Cn_beta (according roskam 2) = 0.0571   1/rad  #ch11, page 25 --> Jan Roskam Part II, equation 11.9
#Since there is no one engine inoperative condition, we size the Cn_dr to be able to maintain a non-yawing flight at a high beta:
#0=Cn_beta*beta+Cn_dr*dr=0 with dr limited to 10 degrees due to using a V-tail. Problematic at low speeds.
#Maximum cross-wind component is given as 36knots at 10m height https://www.smartcockpit.com/docs/Crosswind_Guidelines.pdf  (slide 3)
#Assuming a conventional landing at 1.1Vs,
#) There may be no uncontrollable ground or water looping tendency in 90° cross winds, up to a
#wind velocity of 18.5 km/h (10 knots) at any speed at which the aeroplane may be expected to
#be operated on the ground or water.   https://www.caa.co.uk/media/ippebxfn/caa-cs-vla-amendment-1-initial-airworthiness.pdf page 36
#This is 20 knots for CS25.
#https://www.easa.europa.eu/en/document-library/certification-specifications/group/cs-23-normal-utility-aerobatic-and-commuter-aeroplanes#cs-23-normal-utility-aerobatic-and-commuter-aeroplanes
#Above is for CS23, initial issue , pdf page 300, ground loops --> no requirement
#so take max crosswind as 10 knots as this aircraft is not meant to land normally but it shoud --> 40*1.1=44--> betamax= arctan(10/44) =0.22347
#Cn_dr=-0.0571*0.22347/(10*pi/180)=-0.073


######HARDCODED VALUES
Cn_beta_req=0.0571
Cn_dr_req=-0.073
########################


def get_K(taper_h, AR_h):
    taper_vee=taper_h    #####important, due to CL_alpha_t_h=CL_alpha_N
    AR_vee=AR_h          #####important, due to CL_alpha_t_h=CL_alpha_N
    taper_points = np.array([0.25, 0.5, 1])
    aspect_ratio_points = np.array([3, 10])
    data = np.array([[0.61, 0.64, 0.68], [0.74, 0.77, 0.8]])
    interp_func = RegularGridInterpolator((aspect_ratio_points, taper_points), data)
    AR_interp, taper_interp = np.meshgrid(AR_vee, taper_vee, indexing='ij')
    points_interp = np.stack((AR_interp, taper_interp), axis=-1)
    K = interp_func(points_interp)
    return float(K)

def get_c_control_surface_to_c_vee_ratio(tau):
    ce_c_ratio=np.array([0.05,0.1,0.2,0.3])
    tau_arr=np.array([0.175,0.3,0.47,0.6])
    interp_function=interp1d(tau_arr,ce_c_ratio)
    ce_c_ratio_of_tail=interp_function(tau)
    return float(ce_c_ratio_of_tail)

    
def get_tail_dihedral_and_area(S_hor,Fuselage_volume,S,b,l_v,AR_h,taper_h,Cn_beta_req=0.0571):
    Cn_beta_f=-2*Fuselage_volume/(S*b)    
    K=get_K(taper_h,AR_h)
    S_ver=(Cn_beta_req-Cn_beta_f)/(np.pi/2*AR_h*K)*S*b/l_v   ###Here, the the vertical tail aspect ratio is taken as AR_vee*K = AR_h*K to calculate required vertical tail area
    S_vee=S_ver+S_hor
    v_angle=np.arctan(np.sqrt(S_ver/S_hor))
    return v_angle, S_vee


#YOU ONLY NEED THIS LAST FUNCTION. THE OTHERS ABOVE ARE SUBFUNCTIONS FOR THE NEXT FUNCTION.

def get_control_surface_to_tail_chord_ratio(Lambdah2,b,Fuselage_volume,S_hor,downwash_angle_landing,aoa_landing,CL_h,CL_a_h,V_tail_to_V_ratio,l_v,S,c,taper_h, AR_h,Cn_dr_req=-0.073,beta_h=1,eta_h=0.95,elevator_min=-10*np.pi/180):
    v_angle, S_vee= get_tail_dihedral_and_area(S_hor,Fuselage_volume,S,b,l_v,AR_h,taper_h)
    ##Maximum elevator deflection implicitly limited by suggested design procedure, source 5, fig 1a. 
    CL_tail_de_req=(CL_h-CL_a_h*(aoa_landing-downwash_angle_landing))/elevator_min ###CL_h and CL_a_h comes from the horizontal tail designed.   --> Since we use no angle here, in the next line 
    Cm_de_req_tail=-CL_tail_de_req*(V_tail_to_V_ratio)**2*(S_hor*l_v/(S*c)) ####Get this from CL_de_required, I made this formula --> S_hor or S_vee should be used here
    K = get_K(taper_h,AR_h)
    CL_alpha_N = CLahcalc(AR_h, beta_h, eta_h, Lambdah2)
        
    tau_from_rudder=-Cn_dr_req/(K*CL_alpha_N*np.sin(v_angle)*S_vee/S*l_v/b*V_tail_to_V_ratio**2)   
    tau_from_elevator_requirements=-Cm_de_req_tail/(CL_alpha_N*np.cos(v_angle)*S_vee/S*l_v/c*V_tail_to_V_ratio**2)
    tau=max([tau_from_rudder,tau_from_elevator_requirements])
    ###Recalculate Cm_de or Cn_dr as one of them will now be bigger due to choosing tau as the maximum of the two.
    Cm_de=-V_tail_to_V_ratio**2*tau*l_v/c*CL_alpha_N*S_vee/S*np.cos(v_angle)
    Cn_dr=-V_tail_to_V_ratio**2*tau*l_v/b*K*CL_alpha_N*S_vee/S*np.sin(v_angle)
    print(Cn_dr,Cm_de)
    print(tau)
    c_control_surface_to_c_vee_ratio=get_c_control_surface_to_c_vee_ratio(tau)
    
    return Cm_de, Cn_dr, v_angle, S_vee, c_control_surface_to_c_vee_ratio 
    
