
def get_tail_dihedral_and_area(S_hor,Cn_beta_req,Fuselage_volume,S,b,l_v,AR):
    

    Cn_beta_f=-2*Fuselage_volume/(S*b)
    C_horizontal_to_vee=1  ###Given that aspect ratio of vee is the same as horizontal
    C_vertical_to_vee=1/0.7 ###Given that aspect ratio of vee is the same as vertical tail
    S_ver=(Cn_beta_req-Cn_beta_f)/(-np.pi/2*AR)*S*b/l_v
    S_vee=S_ver+S_hor
    v_angle=np.arctan(np.sqrt(S_ver/S_hor))
    return v_angle, S_vee




#Find a way to get Cn_dr_required
def get_control_surface_to_tail_chord_ratio(v_angle,S_vee,CL_de_required,V_tail_to_V_ratio,l_v,S,c,Cn_dr_req):

    Cm_de_tail= CL_tail_de_req*(V_tail_to_V_ratio)**2*(S_vee*l_v/(S*c)) ####Get this from CL_de_required, I made this formula

    K=            #figure 2 of this paper
    CL_alpha_N=   #figure 3 of reference 7. 
    tau_from_rudder=Cn_dr_req/(K*CL_alpha_N*cos(v_angle)*S_vee/S*l_v/b*V_tail_to_V_ratio**2)
    tau_from_elevator_requirements=Cm_de_req/(CL_alpha_N*cos(v_angle)*S_vee/S*l_v/c*V_tail_to_V_ratio**2)
    tau=max([tau_from_rudder,tau_from_elevator_requirements])
    ###Recalculate Cm_de or Cn_dr as one of them will now be bigger due to choosing tau as the maximum of the two.

    Cm_de=-V_tail_to_V_ratio**2*tau*l_v/c*CL_alpha_N*S_vee/S*cos(v_angle)
    Cn_dr=-V_tail_to_V_ratio**2*tau*l_v/b*K*CL_alpha_N*S_vee/S*cos(v_angle)
    c_control_surface_to_c_vee_ratio= #step 8

    return Cm_de, Cn_dr, c_control_surface_to_c_vee_ratio 
    
