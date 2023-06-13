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


###Lower taper results more surface area but the minimum is limited by structural integrity.

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
    ce_c_ratio=np.array([0,0.15,0.3])
    tau_arr=np.array([0,0.35,0.55])
    interp_function=interp1d(tau_arr,ce_c_ratio)
    ce_c_ratio_of_tail=interp_function(tau)
    return float(ce_c_ratio_of_tail)

    
def get_tail_dihedral_and_area(Lambdah2,S_hor,Fuselage_volume,S,b,l_v,AR_h,taper_h,Cn_beta_req=0.0571,beta_h=1,eta_h=0.95):
    Cn_beta_f=-2*Fuselage_volume/(S*b)    
    K=get_K(taper_h,AR_h)
    CL_alpha_N = CLahcalc(AR_h, beta_h, eta_h, Lambdah2)
    S_ver=(Cn_beta_req-Cn_beta_f)/(K*CL_alpha_N)*S*b/l_v   ###Here, the the vertical tail aspect ratio is taken as AR_vee*K = AR_h*K to calculate required vertical tail area
    S_vee=S_ver+S_hor
    v_angle=np.arctan(np.sqrt(S_ver/S_hor))
    return v_angle, S_vee


#YOU ONLY NEED THIS LAST FUNCTION. THE OTHERS ABOVE ARE SUBFUNCTIONS FOR THE NEXT FUNCTION.

def get_control_surface_to_tail_chord_ratio(Wing, Fuse, HorTail,Aero,  CL_h,l_v,Cn_beta_req=0.0571,beta_h=1,eta_h=0.95,total_deflection=20*np.pi/180,design_cross_wind_speed=9,step=0.1*np.pi/180,axial_induction_factor=0.005):
    V_stall = Aero.v_stall
    Lambdah2 = HorTail.sweep_halfchord_h
    b = Wing.span
    Fuselage_volume = Fuse.volume_fuselage
    S_hor = HorTail.surface
    downwash_angle_landing = HorTail.downwash_angle
    aoa_landing = Aero.alpha_approach
    Vh_V2 = 0.95*(1+axial_induction_factor)**2 #assumed
    S = Wing.surface
    c = Wing.chord_mac
    taper_h = HorTail.taper_h
    AR_h = HorTail.aspect_ratio
    

    tau_from_rudder=0     ##Just to initialize loop
    tau_from_elevator=1   ##Just to initialize loop
    elevator_min=-1*np.pi/180
    rudder_max=total_deflection+elevator_min
    v_angle, S_vee= get_tail_dihedral_and_area(Lambdah2,S_hor,Fuselage_volume,S,b,l_v,AR_h,taper_h)
    K = get_K(taper_h,AR_h)
    CL_alpha_N = CLahcalc(AR_h, beta_h, eta_h, Lambdah2)
    
    while (tau_from_elevator>tau_from_rudder and rudder_max>1*np.pi/180):
                
        Cn_dr_req=-Cn_beta_req*np.arctan(design_cross_wind_speed/V_stall)/(rudder_max)
        CL_tail_de_req=(CL_h-CL_alpha_N*(aoa_landing-downwash_angle_landing))/elevator_min
        print(Cn_dr_req)
    ###CL_h and CL_a_h comes from the horizontal tail designed.   --> Since we use no angle here, in the next line 
        Cm_de_req_tail=-CL_tail_de_req*(Vh_V2)*(S_hor*l_v/(S*c)) ####Get this from CL_de_required, I made this formula --> S_hor or S_vee should be used here
                
        tau_from_rudder=-Cn_dr_req/(K*CL_alpha_N*np.sin(v_angle)*S_vee/S*l_v/b*Vh_V2)
        tau_from_elevator=-Cm_de_req_tail/(CL_alpha_N*np.cos(v_angle)*S_vee/S*l_v/c*Vh_V2)
        print(tau_from_rudder)
        elevator_min=elevator_min-step
        rudder_max=rudder_max-step        
        #print(tau_from_rudder,tau_from_elevator)

    tau=max([tau_from_rudder,tau_from_elevator])
    ###Recalculate Cm_de or Cn_dr as one of them will now be bigger due to choosing tau as the maximum of the two.
    Cm_de=-Vh_V2*tau*l_v/c*CL_alpha_N*S_vee/S*np.cos(v_angle)
    Cn_dr=-Vh_V2*tau*l_v/b*K*CL_alpha_N*S_vee/S*np.sin(v_angle)
        
    if tau>0.55:
        c_control_surface_to_c_vee_ratio="Not possible. Lower CLh in the horizontal tail sizing program"
    else:
        c_control_surface_to_c_vee_ratio=get_c_control_surface_to_c_vee_ratio(tau)
    
    return [np.degrees(rudder_max),np.degrees(elevator_min), tau, Cm_de, Cn_dr, v_angle, S_vee, c_control_surface_to_c_vee_ratio]


if __name__ == "__main__":
    rudder, elevator, tau, Cm_de, Cn_dr, v_angle, S_vee, c_control_surface_to_c_vee_ratio = get_control_surface_to_tail_chord_ratio(V_stall=44,Lambdah2=0,b=10,Fuselage_volume=10,S_hor=1.8897,downwash_angle_landing=0.14,aoa_landing=0.2,CL_h=-0.5,CL_a_h=5,V_tail_to_V_ratio=0.9,l_v=5,S=12,c=1.5,taper_h=1, AR_h=6)
    print(v_angle, S_vee, tau,c_control_surface_to_c_vee_ratio, Cm_de, Cn_dr, v_angle, S_vee, c_control_surface_to_c_vee_ratio)
 
