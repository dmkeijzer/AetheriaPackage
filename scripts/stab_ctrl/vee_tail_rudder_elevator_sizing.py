from scipy.interpolate import RegularGridInterpolator,interp1d
import numpy as np
from wing_loc_horzstab_sizing import CLahcalc

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
    return K

def get_c_control_surface_to_c_vee_ratio(tau):
    ce_c_ratio=np.array([0.05,0.1,0.2,0.3])
    tau_arr=np.array([0.175,0.3,0.47,0.6])
    interp_function=interp1d(tau_arr,ce_c_ratio)
    ce_c_ratio_of_tail=interp_function(tau)
    return ce_c_ratio_of_tail

    
def get_tail_dihedral_and_area(S_hor,Cn_beta_req,Fuselage_volume,S,b,l_v,AR_h,taper_h):
    Cn_beta_f=-2*Fuselage_volume/(S*b)    
    K=get_K(taper_h,AR_h)
    S_ver=(Cn_beta_req-Cn_beta_f)/(-np.pi/2*AR_h*K)*S*b/l_v   ###Here, the the vertical tail aspect ratio is taken as AR_vee*K = AR_h*K to calculate required vertical tail area
    S_vee=S_ver+S_hor
    v_angle=np.arctan(np.sqrt(S_ver/S_hor))
    return v_angle, S_vee


#Find a way to get Cn_dr_required
def get_control_surface_to_tail_chord_ratio(downwash_angle_landing,aoa_landing,CL_h, CL_a_h,S_vee,V_tail_to_V_ratio,l_v,S,c,taper_h, AR_h,beta_h,eta_h,Lambdah2, Cn_dr_req,v_angle):
    elevator_min=-10*np.pi/180 ##Maximum elevator deflection implicitly limited by suggested design procedure, source 5, fig 1a. 
    CL_tail_de_required=(CL_h-CL_a_h*(aoa_landing-downwash_angle_landing))/elevator_min ###CL_h and CL_a_h comes from the horizontal tail designed.
    Cm_de_req_tail=CL_tail_de_req*(V_tail_to_V_ratio)**2*(S_vee*l_v/(S*c)) ####Get this from CL_de_required, I made this formula
    K=get_K(taper_h,AR_h)
    CL_alpha_N= CLahcalc(AR_h, beta_h, eta_h, Lambdah2) 
    tau_from_rudder=Cn_dr_req/(K*CL_alpha_N*cos(v_angle)*S_vee/S*l_v/b*V_tail_to_V_ratio**2)
    tau_from_elevator_requirements=Cm_de_req_tail/(CL_alpha_N*cos(v_angle)*S_vee/S*l_v/c*V_tail_to_V_ratio**2)
    tau=max([tau_from_rudder,tau_from_elevator_requirements])
    ###Recalculate Cm_de or Cn_dr as one of them will now be bigger due to choosing tau as the maximum of the two.
    Cm_de=-V_tail_to_V_ratio**2*tau*l_v/c*CL_alpha_N*S_vee/S*cos(v_angle)
    Cn_dr=-V_tail_to_V_ratio**2*tau*l_v/b*K*CL_alpha_N*S_vee/S*cos(v_angle)
    c_control_surface_to_c_vee_ratio=get_c_control_surface_to_c_vee_ratio(tau)
    
    return Cm_de, Cn_dr, c_control_surface_to_c_vee_ratio 
    
